"""观察与公开问题入口共用的私有评分字段／凭据隔离。"""
PRIVATE_FIELDS = frozenset({
    "gt", "answer", "ground_truth", "ground_truth_json", "correct_answer",
    "reference_answer", "expected_answer", "api_key", "authorization",
    "access_token", "password",
})
QUESTION_FIELDS = frozenset({"question", "questions", "options", "response_options"})


def reject_private(value, *, observation=False):
    forbidden = PRIVATE_FIELDS | (QUESTION_FIELDS if observation else frozenset())
    if isinstance(value, dict):
        if any(str(key).strip().lower() in forbidden for key in value):
            raise ValueError("private scoring fields, credentials or disallowed questions in input")
        for item in value.values():
            reject_private(item, observation=observation)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_private(item, observation=observation)
