"use strict";
const $ = (id) => document.getElementById(id);
let accessKey = "";
let snapshot = null;
let refreshing = false;
let generation = 0;
const states = {
  model_staged: ["等待启用", "模型文件已就位，等待推理引擎启动"],
  waiting_for_model: ["等待模型", "控制台已运行，推理尚未启用"],
  engine_unavailable: ["引擎未就绪", "请检查模型挂载与内部服务状态"],
  ready: ["运行就绪", "已准备好接收新的推理会话"],
  draining: ["已暂停新会话", "已有会话可以继续或正常结束"],
};
const eventNames = {
  node_started: "节点已启动", admission_opened: "已恢复接收新会话",
  admission_paused: "已暂停新会话", session_opened: "推理会话已创建",
  session_closed: "推理会话已关闭", request_rejected: "请求未被接收",
  request_timeout: "请求超时", engine_unreachable: "推理引擎连接失败",
  engine_response_invalid: "推理响应校验失败",
};
const errors = {
  invalid_access_key: "访问密钥不正确，请重新输入。",
  access_key_not_configured: "节点尚未配置访问密钥，请先执行部署初始化。",
  invalid_request: "请求未被接受，请重试。",
};
$("endpoint").value = location.origin;
function notice(message, isError = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", isError);
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {Authorization: `Bearer ${accessKey}`, "Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store", signal: AbortSignal.timeout(10000),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(errors[value.error?.code] || "节点连接失败，请检查服务状态。");
  return value;
}
function step(id, done, pending) {
  const item = $(id);
  item.classList.toggle("done", done);
  item.querySelector(".step-status").textContent = done ? "已完成" : pending;
}
function render(value) {
  snapshot = value;
  const [title, note] = states[value.state] || ["状态未知", "请刷新节点状态"];
  $("service-state").textContent = title;
  $("service-note").textContent = note;
  $("service-dot").className = `dot ${value.ready ? "ok" : "warn"}`;
  $("gpu-state").textContent = value.gpu_ready ? "推理已就绪" : "未就绪";
  $("gpu-note").textContent = value.gpu_ready ? "引擎已完成设备与模型初始化" : "模型初始化成功后才启用推理";
  $("model-state").textContent = value.checkpoint.present ? "已挂载" : "尚未挂载";
  $("model-note").textContent = value.checkpoint.present ? `${(value.checkpoint.bytes / 1073741824).toFixed(2)} GiB · 本地文件` : "等待接入本地权重文件";
  $("mode-label").textContent = value.mode === "framework" ? "框架模式" : "独立推理节点";
  $("uptime").textContent = `${Math.floor(value.uptime_seconds / 3600)} 小时 ${Math.floor(value.uptime_seconds % 3600 / 60)} 分钟`;
  $("request-count").textContent = `${value.requests.completed} / ${value.requests.failed}`;
  $("active-session").textContent = Number.isInteger(value.active_sessions)
    ? `${value.active_sessions} / ${value.camera_capacity || 1} 路使用中`
    : value.active_session === null ? "尚未读取" : value.active_session ? "会话进行中" : "空闲";
  $("admission-button").textContent = value.accepting_sessions ? "暂停新会话" : "恢复接收新会话";
  $("refresh-button").disabled = false;
  $("admission-button").disabled = false;
  step("step-framework", true, "待检查");
  step("step-model", value.checkpoint.present, "等待接入");
  step("step-ready", value.gpu_ready, "等待初始化");
  const events = $("events");
  events.replaceChildren();
  for (const row of value.events.slice(0, 4)) {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = new Date(row.time * 1000).toLocaleTimeString("zh-CN", {hour12: false});
    item.append(time, document.createTextNode(eventNames[row.code] || "节点状态已更新"));
    events.append(item);
  }
}
function renderMemory(value) {
  const bytes = (n) => `${(n / 1048576).toFixed(0)} MiB`;
  const available = value?.available === true;
  $("memory-state").textContent = available ? "实时读取" : "暂不可用";
  $("memory-allocated").textContent = available ? bytes(value.allocated_bytes) : "—";
  $("memory-cached").textContent = available ? bytes(value.unused_reserved_bytes) : "—";
  $("memory-free").textContent = available ? `${bytes(value.device_free_bytes)} / ${bytes(value.device_total_bytes)}` : "—";
  $("memory-note").textContent = !available ? "等待引擎提供显存数据。" : value.error_code
    ? "上次回收未成功，将在空闲时重试。"
    : value.reclaim_pending ? "等待所有会话结束后回收缓存。"
    : value.last_reclaim ? `最近回收 ${bytes(value.last_reclaim.released_bytes)} · ${new Date(value.last_reclaim.time * 1000).toLocaleTimeString("zh-CN", {hour12: false})}`
    : "全部会话结束后，自动回收空闲缓存。";
}
async function refresh() {
  if (!accessKey || refreshing) return false;
  refreshing = true;
  const current = generation;
  try {
    const [result, memory] = await Promise.all([
      api("/api/status"), api("/api/gpu-memory").catch(() => null),
    ]);
    if (current !== generation) return false;
    render(result);
    renderMemory(memory);
    $("connect-form").hidden = true;
    $("disconnect-button").hidden = false;
    $("connection-hint").textContent = "已登录节点控制台。状态每 5 秒自动更新，密钥未写入浏览器存储。主程序仍需单独配置节点地址和密钥。";
    notice(result.mode === "framework" ? (result.checkpoint.present ? "模型文件已就位。启用 GPU 运行服务后开始推理。" : "框架已就绪。接入真实模型后才会提供推理结果。") : "状态已更新。");
    return true;
  } catch (error) {
    if (current !== generation) return false;
    renderMemory(null);
    notice(error.message || "节点连接失败。", true);
    $("service-state").textContent = "连接中断";
    $("service-note").textContent = "其余数据为上次读取结果";
    $("service-dot").className = "dot warn";
    $("admission-button").disabled = true;
    return false;
  } finally { refreshing = false; }
}
$("connect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  accessKey = $("access-key").value.trim();
  $("connect-button").disabled = true;
  notice("正在验证登录密钥…");
  if (await refresh()) $("access-key").value = "";
  else accessKey = "";
  $("connect-button").disabled = false;
});
$("disconnect-button").addEventListener("click", () => {
  accessKey = ""; generation++; snapshot = null;
  location.reload();
});
$("refresh-button").addEventListener("click", refresh);
$("admission-button").addEventListener("click", async () => {
  if (!snapshot) return;
  $("admission-button").disabled = true;
  try {
    await api("/api/admission", {accepting_sessions: !snapshot.accepting_sessions});
    await refresh();
  } catch (error) { notice(error.message, true); }
  finally { $("admission-button").disabled = !accessKey; }
});
$("copy-button").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("endpoint").value); notice("节点地址已复制。"); }
  catch { $("endpoint").select(); notice("请复制选中的节点地址。"); }
});
setInterval(refresh, 5000);
