const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

function canvas() {
  const context = {
    clearRect() {}, drawImage() {}, setLineDash() {}, strokeRect() {}, fillText() {},
  };
  return {width:320, height:180, clientWidth:320, clientHeight:180,
    getContext:() => context, toDataURL:() => "data:image/jpeg;base64,eA=="};
}

function trackerView() {
  const listeners = new Map();
  const video = {videoWidth:320, videoHeight:180, seeking:false,
    addEventListener:(name, callback) => listeners.set(name, callback),
    requestVideoFrameCallback:() => 1, cancelVideoFrameCallback() {}};
  const context = {window:{previewDetector:true, currentStage:"stage", trackingClient:"client"},
    document:{createElement:canvas}, performance:{now:() => 1000},
    AbortController, setTimeout:() => 1, clearTimeout() {},
    fetch:() => {throw new Error("unexpected detector request");}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,
    "../tools/vlm_benchmark/preview_tracking.js"), "utf8"), context);
  const Tracker = context.window.PreviewTracking;
  const tracker = new Tracker(video, canvas(), {}, 0, {duration:10});
  tracker.playing = true;
  const paints = [];
  tracker.paint = (image, objects=[]) => {paints.push(objects);};
  return {Tracker, tracker, paints, listeners, context};
}

const oldBox = {track_id:"old-person", bbox:{x1:.1, y1:.1, x2:.5, y2:.9}, event_state:"alert"};

test("budgeted detector lanes repaint current frames and expire held boxes", async () => {
  const {Tracker, tracker, paints} = trackerView();
  Tracker.inflight = Tracker.maxInflight = 4;
  tracker.lastObjects = [oldBox];
  tracker.lastObjectsAt = 1000;
  tracker.lastPaint = 1000;
  await tracker.frame(2000, {mediaTime:1});
  assert.equal(paints.length, 1);
  assert.equal(paints[0][0], oldBox);
  await tracker.frame(6000, {mediaTime:2});
  assert.equal(paints.length, 2);
  assert.equal(paints[1].length, 0, "expired boxes must disappear even with no request slot");
  assert.equal(tracker.stats.budgetSkipped, 2);
  assert.equal(tracker.stats.requests, 0);
});

test("seeking clears held detector boxes before the next busy frame", async () => {
  const {tracker, paints, listeners} = trackerView();
  tracker.lastObjects = [oldBox];
  tracker.lastObjectsAt = 1000;
  tracker.busy = true;
  listeners.get("seeking")();
  await tracker.frame(1100, {mediaTime:5});
  assert.equal(tracker.generation, 1);
  assert.equal(paints.length, 1);
  assert.equal(paints[0].length, 0, "a pre-seek box must not reappear");
});

test("clip wraps invalidate detector boxes and pending response generations", async () => {
  const {tracker, paints} = trackerView();
  tracker.lastObjects = [oldBox];
  tracker.lastObjectsAt = 1000;
  tracker.lastTime = 9.9;
  tracker.busy = true;
  await tracker.frame(1100, {mediaTime:0.1});
  assert.equal(tracker.generation, 1);
  assert.equal(tracker.epoch, 1);
  assert.equal(tracker.lastTime, 10.1);
  assert.equal(paints[0].length, 0, "a previous loop's actor must not be reused");
});

test("a detector response from the previous clip loop is discarded", async () => {
  const {tracker, paints, context} = trackerView();
  let complete;
  context.fetch = () => new Promise(resolve => {complete = resolve;});
  const request = tracker.frame(1000, {mediaTime:9.9});
  await tracker.frame(1100, {mediaTime:.1});
  complete({ok:true, json:async () => ({event_state:"alert", status:"tracking", objects:[oldBox]})});
  await request;
  assert.equal(tracker.epoch, 1);
  assert.equal(tracker.stats.staleResponses, 1);
  assert.equal(tracker.lastObjects.length, 0);
  assert(paints.every(objects => objects.length === 0));
  let payload;
  context.fetch = async (_url, options) => {
    payload = JSON.parse(options.body);
    return {ok:true, json:async () => ({event_state:"normal", status:"no_person_boxes", objects:[]})};
  };
  await tracker.frame(1200, {mediaTime:.2});
  assert.equal(payload.epoch, 1);
  assert.equal(payload.time, 10.2);
});

test("tracking payload generation changes on forward seeks and stays stable during playback", async () => {
  const {tracker, context, listeners} = trackerView();
  const payloads = [];
  context.fetch = async (_url, options) => {
    payloads.push(JSON.parse(options.body));
    return {ok:true, json:async () => ({event_state:"normal", status:"no_person_boxes", objects:[]})};
  };
  await tracker.frame(1000, {mediaTime:1});
  await tracker.frame(1100, {mediaTime:2});
  listeners.get("seeking")();
  await tracker.frame(1200, {mediaTime:5});
  await tracker.frame(1300, {mediaTime:6});
  assert.deepEqual(payloads.map(p=>p.generation), [0,0,1,1]);
  assert.deepEqual(payloads.map(p=>p.epoch), [0,0,0,0]);
  assert.deepEqual(payloads.map(p=>p.time), [1,2,5,6]);
});

test("all 12 detector lanes progress before the first lanes reacquire slots", async () => {
  const {Tracker, tracker, context} = trackerView();
  const lanes = [tracker];
  for(let camera=1; camera<12; camera++) {
    const next = new Tracker(tracker.video, canvas(), {}, camera, {duration:10});
    next.playing = true;
    lanes.push(next);
  }
  const requests = [];
  context.fetch = (url) => new Promise(resolve => {requests.push({camera:Number(url.split("/").at(-1)), resolve});});
  const response = {ok:true, json:async () => ({event_state:"normal", status:"no_person_boxes", objects:[]})};
  let pending = lanes.map(lane => lane.frame(1000, {mediaTime:1}));
  assert.deepEqual(requests.map(r=>r.camera), [0,1,2,3]);
  requests.slice(0,4).forEach(r=>r.resolve(response));
  await Promise.all(pending);
  await Promise.all(lanes.slice(0,4).map(lane=>lane.frame(1100, {mediaTime:2})));
  assert.equal(requests.length, 4, "completed lanes wait behind queued lanes");
  pending = lanes.slice(4,8).map(lane=>lane.frame(1100, {mediaTime:2}));
  assert.deepEqual(requests.map(r=>r.camera), [0,1,2,3,4,5,6,7]);
  requests.slice(4,8).forEach(r=>r.resolve(response));
  await Promise.all(pending);
  await Promise.all(lanes.slice(0,4).map(lane=>lane.frame(1200, {mediaTime:3})));
  assert.equal(requests.length, 8);
  pending = lanes.slice(8,12).map(lane=>lane.frame(1200, {mediaTime:3}));
  assert.deepEqual(requests.map(r=>r.camera), Array.from({length:12}, (_,i)=>i));
  requests.slice(8,12).forEach(r=>r.resolve(response));
  await Promise.all(pending);
  assert.equal(Tracker.inflight, 0);
});

for(const inactive of ["closed", "paused"]) {
  test(`a ${inactive} detector waiter cannot block another lane`, async () => {
    const {Tracker, tracker, context} = trackerView();
    Tracker.inflight = Tracker.maxInflight = 1;
    await tracker.frame(1000, {mediaTime:1});
    if(inactive === "closed") tracker.close();
    else tracker.playing = false;
    const next = new Tracker(tracker.video, canvas(), {}, 1, {duration:10});
    next.playing = true;
    Tracker.inflight = 0;
    context.fetch = async () => ({ok:true, json:async () => ({event_state:"normal", status:"no_person_boxes", objects:[]})});
    await next.frame(1100, {mediaTime:2});
    assert.equal(next.stats.requests, 1);
    assert.equal(Tracker.inflight, 0);
  });
}

test("a detector waiter whose media stops producing frames cannot block a live lane", async () => {
  const {Tracker, tracker, context} = trackerView();
  Tracker.inflight = Tracker.maxInflight = 1;
  await tracker.frame(1000, {mediaTime:1});
  const live = new Tracker(tracker.video, canvas(), {}, 1, {duration:10});
  live.playing = true;
  await live.frame(1500, {mediaTime:1});
  assert.equal(Tracker.waiters.length, 2);
  Tracker.inflight = 0;
  context.fetch = async () => ({ok:true, json:async () => ({event_state:"normal", status:"no_person_boxes", objects:[]})});
  await live.frame(2101, {mediaTime:2});
  assert.equal(tracker.playing, true, "the stalled lane still claims to be playing");
  assert.equal(live.stats.requests, 1);
  assert.equal(Tracker.waiters.length, 0);
});

function element(tag="div") {
  const value = {tag, children:[], style:{setProperty() {}}, dataset:{},
    classList:{toggle() {}}, textContent:"", hidden:false,
    append(...children) {this.children.push(...children);},
    replaceChildren(...children) {this.children = children;},
    addEventListener() {}, removeAttribute() {}, load() {},
    getBoundingClientRect:() => ({top:0, left:0, bottom:10, right:10})};
  Object.defineProperty(value, "firstChild", {get:() => value.children[0]});
  if(tag === "canvas") Object.assign(value, canvas());
  if(tag === "video") Object.assign(value, {readyState:2, paused:true, currentTime:0,
    getVideoPlaybackQuality:() => ({totalVideoFrames:10, droppedVideoFrames:0}),
    play() {this.paused=false; return Promise.resolve();},
    pause() {this.paused=true;}});
  return value;
}

async function monitorView(tracking) {
  const elements = new Map(), videos = [], animations = [], timers = [];
  const state = {stage:"stage", state:"playing", detector:true, monitor_layout:"evidence",
    cameras:[{camera:1, tick:-1, lanes:{}, rejected:0, out_of_order:0}],
    media:[{url:"preview.mp4", duration:10, sample_count:2, timestamps:[0,5]}],
    lanes:["events"], mode:"events", phase:"measurement", model:"model", backend:"local",
    real:true, repeat:1, planned:1, counters:{events:{complete:0}}, source_count:1,
    observation_fps:1, elapsed:1, offsets:[0], duration:10, sample_fps:.2,
    summary:{input_timing_valid:true, all_drained:true, result_records_missing:0}};
  const document = {body:element("body"),
    getElementById(id) {if(!elements.has(id)) elements.set(id, element()); return elements.get(id);},
    createElement(tag) {const value=element(tag); if(tag==="video") videos.push(value); return value;},
    querySelectorAll:() => []};
  const context = {window:{}, document, crypto:{randomUUID:() => "client"},
    location:{search:tracking ? "?tracking=on" : ""}, URLSearchParams,
    performance:{now:() => 1000}, innerHeight:1080, innerWidth:1920,
    requestAnimationFrame:callback => {animations.push(callback);},
    setTimeout:callback => {timers.push(callback);},
    fetch:async () => ({ok:true, json:async () => state})};
  let tracker;
  context.window.PreviewTracking = class {constructor() {tracker = this; this.epoch=0;} close() {}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,
    "../tools/vlm_benchmark/monitor.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  function animate() {assert(animations.length); animations.shift()();}
  animate();
  return {state, video:videos[0], clock:elements.get("grid").children[0].children[0].children[1],
    get tracker() {return tracker;}, animate, async poll() {
    assert(timers.length); await timers.shift()();
  }};
}

test("the monitor preserves each detector tracker's actual source-loop epoch", async () => {
  const view = await monitorView(true);
  view.tracker.epoch = 3;
  view.animate();
  assert.equal(view.tracker.epoch, 3);
});

test("the detector source clock displays the frame's actual playback time", async () => {
  const view = await monitorView(true);
  view.video.currentTime = 7.25;
  view.animate();
  assert.match(view.clock.textContent, /^视频 7\.25 s ·/);
});

for(const tracking of [false, true]) {
  for(const terminal of ["draining", "finished", "failed"]) {
    test(`detector playback pauses on ${terminal} with tracking ${tracking ? "on" : "off"}`, async () => {
      const view = await monitorView(tracking);
      assert.equal(view.video.paused, false);
      view.state.state = terminal;
      await view.poll();
      view.animate();
      assert.equal(view.video.paused, true);
    });
  }
}
