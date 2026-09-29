(function (root) {
  "use strict";
  function resultText(row) {
    if (row.response_outcome === "failed") return "请求失败 / " + (row.failure_category || "unknown");
    if (row.response_outcome !== "validated") return "等待响应";
    const positive = row.detected_labels || [];
    const missing = row.missing_event_labels || [];
    const unknown = row.unknown_event_labels || [];
    if (missing.length || unknown.length || row.result === "partial") {
      return "部分完成 / 无法判断" + (positive.length ? "；阳性：" + positive.join(", ") : "") +
        (missing.length ? "；缺失：" + missing.join(", ") : "") +
        (unknown.length ? "；不确定：" + unknown.join(", ") : "");
    }
    return positive.length ? "阳性：" + positive.join(", ") : "已返回判定均为阴性（不代表所有人物）";
  }
  function diagnosticsBase(cameraId) {
    return /^camera[0-9]{2,}$/.test(cameraId || "") ?
      "/v1/fleet/cameras/" + cameraId + "/scene/diagnostics" : "/v1/scene/diagnostics";
  }
  if (typeof module !== "undefined") module.exports = {resultText, diagnosticsBase};
  if (!root.document) return;
  const panel = document.getElementById("scene-diagnostics");
  if (!panel) return;
  const content = document.getElementById("scene-diagnostics-content");
  const base = diagnosticsBase(root.GoodvisionDiagnosticCameraId);
  let timer, controller, generation = 0;
  function stop() { generation++; clearTimeout(timer); controller?.abort(); }
  async function refresh(version) {
    if (!panel.open || document.hidden || version !== generation) return;
    let poll = true;
    const requestController = new AbortController();
    controller = requestController;
    const timeout = setTimeout(() => requestController.abort(), 3000);
    try {
      const response = await fetch(base, {cache:"no-store", signal:requestController.signal});
      if (!response.ok) throw new Error("unavailable");
      const data = await response.json();
      if (version !== generation) return;
      content.replaceChildren();
      if (data.enabled === false) {
        poll = false;
        content.textContent = "诊断未启用。";
        return;
      }
      const stats = document.createElement("pre");
      stats.textContent = JSON.stringify({scope:data.scope, delivery:data.delivery, counts:data.counts, failures:data.failures,
        unanalyzed_gaps:data.unanalyzed_gaps, latency_seconds:data.latency_seconds}, null, 2);
      content.append(stats);
      for (const row of (data.records || []).slice(0, 8)) {
        const item = document.createElement("div"), text = document.createElement("p");
        text.textContent = resultText(row) + " · 证据：" + row.evidence_outcome +
          (row.confirmed_event_labels === null ? " · 事件确认：此诊断未采集" :
            " · 已确认事件：" + ((row.confirmed_event_labels || []).join(", ") || "无")) +
          (row.output_mode ? " · 模型通道：" + row.output_mode : "") +
          " · 源时间：" + row.source_start_seconds + "–" + row.source_end_seconds + "s";
        item.append(text);
        if (row.images_available) for (const frame of row.frames || []) {
          const figure = document.createElement("figure"), img = document.createElement("img");
          const caption = document.createElement("figcaption");
          img.src = base + "/" + encodeURIComponent(row.request_id) + "/" + frame.frame_sequence + ".jpg";
          img.alt = "实际请求图片"; img.loading = "lazy"; img.style.maxWidth = "240px";
          caption.textContent = "帧 " + frame.frame_sequence + " · " + frame.timestamp_seconds +
            "s · " + frame.width + "×" + frame.height;
          figure.append(img, caption); item.append(figure);
        }
        content.append(item);
      }
    } catch (_) {
      if (version === generation) content.textContent = "诊断暂不可用；需要 active 连续场景会话。";
    } finally {
      clearTimeout(timeout);
      if (poll && version === generation && panel.open && !document.hidden)
        timer = setTimeout(() => refresh(version), 2000);
    }
  }
  function restart() { stop(); refresh(generation); }
  panel.addEventListener("toggle", restart);
  document.addEventListener("visibilitychange", restart);
  root.addEventListener("beforeunload", stop);
})(globalThis);
