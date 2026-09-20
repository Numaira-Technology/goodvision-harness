"""JPEG predecode must preserve process warning policy under concurrency."""

import io
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi import HTTPException
from PIL import Image

from workers.sam_scene.app import Settings, decode_jpeg


def _jpeg():
    output = io.BytesIO()
    with Image.new("RGB", (64, 40), (80, 100, 120)) as image:
        image.save(output, format="JPEG")
    return output.getvalue()


@pytest.mark.parametrize("failure", [None, "open", "convert"])
def test_interleaved_decoders_never_change_global_warning_filters(monkeypatch, failure):
    payload = _jpeg()
    original_open, original_convert = Image.open, Image.Image.convert
    filters = warnings.filters
    filters_before = list(filters)
    first_inside, second_inside, first_finished = (threading.Event() for _ in range(3))
    thread_state = threading.local()
    observations = []

    def observe():
        observations.append(
            warnings.filters is filters and warnings.filters == filters_before
        )

    def ordered_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        try:
            observe()
            if thread_state.index == 0:
                first_inside.set()
                assert second_inside.wait(3)
            else:
                second_inside.set()
                assert first_finished.wait(3)
                if failure == "open":
                    raise OSError("private decoder detail")
            return image
        except BaseException:
            image.close()
            raise

    def convert(*args, **kwargs):
        observe()
        if thread_state.index == 1 and failure == "convert":
            raise OSError("private decoder detail")
        return original_convert(*args, **kwargs)

    def decode(index):
        thread_state.index = index
        try:
            try:
                image = decode_jpeg(payload, Settings())
            except HTTPException as error:
                assert "private" not in error.detail
                return error.status_code
            with image:
                assert image.size == (64, 40) and image.mode == "RGB"
            return 200
        finally:
            if index == 0:
                first_finished.set()

    monkeypatch.setattr(Image, "open", ordered_open)
    monkeypatch.setattr(Image.Image, "convert", convert)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(decode, 0)
            assert first_inside.wait(3)
            second = executor.submit(decode, 1)
            assert first.result(timeout=5) == 200
            assert second.result(timeout=5) == (400 if failure else 200)
        assert warnings.filters is filters
        assert warnings.filters == filters_before
        # Also catch transient mutations even if exit order happens to restore
        # the final value. The other lane must see the original policy throughout.
        assert observations and all(observations)
    finally:
        # Keep a failing regression against the old decoder isolated from other tests.
        warnings.filters = filters
        filters[:] = filters_before


@pytest.mark.parametrize("warning_policy", ["ignore", "default", "error"])
@pytest.mark.parametrize(
    "pillow_limit,pixel_limit,expected",
    [
        (None, 2560, 200),
        (None, 2559, 413),
        (2560, 8294400, 200),
        (1500, 8294400, 413),  # Pillow warning threshold, not its hard error limit.
        (1000, 8294400, 413),  # Pillow hard limit, before Image.open returns.
        (4000, 2559, 413),
    ],
)
def test_pixel_guards_precede_rgb_allocation_and_do_not_depend_on_warning_policy(
    monkeypatch, warning_policy, pillow_limit, pixel_limit, expected
):
    payload = _jpeg()
    original_convert = Image.Image.convert
    allocations = []

    def convert(*args, **kwargs):
        allocations.append(True)
        return original_convert(*args, **kwargs)

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    monkeypatch.setattr(Image.Image, "convert", convert)
    settings = replace(Settings(), max_image_pixels=pixel_limit)
    # Test setup alone owns this context; no other thread modifies warning state.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter(warning_policy, Image.DecompressionBombWarning)
        filters = warnings.filters
        filters_before = list(filters)
        if expected == 200:
            with decode_jpeg(payload, settings) as image:
                assert image.size == (64, 40)
            assert allocations == [True]
        else:
            with pytest.raises(HTTPException) as error:
                decode_jpeg(payload, settings)
            assert error.value.status_code == expected
            assert not allocations
        assert warnings.filters is filters and warnings.filters == filters_before
