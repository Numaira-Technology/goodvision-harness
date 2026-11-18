// Actual page JavaScript, synthetic DOM/network/clock: no browser, media or models.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const code = fs.readFileSync('tools/bag_demo/page.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1];
const tick = () => new Promise(resolve => setImmediate(resolve));
const response = data => ({ok:true, json:async () => data});
function deferred() {
  let resolve;
  const promise = new Promise(done => {resolve = done;});
  return {promise, resolve};
}
async function page() {
  const nodes = new Map(), timers = new Map(), lifecycle = {};
  let timerID = 0;
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, {
      textContent:'', innerHTML:'', style:{}, dataset:{}, handlers:{},
      classList:{toggle() {}}, appendChild() {}, load() {},
      paused:true, ended:false, currentTime:1, checked:false,
      addEventListener(name, fn) {this.handlers[name] = fn;},
      getContext() {return {};},
    });
    return nodes.get(id);
  };
  const context = vm.createContext({console, AbortController,
    requestAnimationFrame() {},
    setTimeout(fn, ms) {assert.ok(ms > 0 && ms <= 60000); timers.set(++timerID, fn); return timerID;},
    clearTimeout(id) {timers.delete(id);},
    window:{addEventListener(name, fn) {lifecycle[name] = fn;}},
    document:{querySelector:element, querySelectorAll:() => [], createElement:element},
    fetch:async () => response({clips:[]}),
  });
  vm.runInContext(code, context);
  const run = text => vm.runInContext(text, context);
  await tick();
  context.fetch = async () => response({frames:[]});
  await run("selectClip({name:'a.MOV',truth:null})");
  const requests = [];
  context.fetch = (url, options) => {
    if (!url.startsWith('/api/verdict')) return Promise.resolve(response({frames:[]}));
    const pending = deferred();
    requests.push({url, ...pending, signal:options?.signal});
    // Deliberately ignore abort: late settlements must still be harmless.
    return pending.promise;
  };
  return {run, context, element, requests, timers, lifecycle,
    start(t) {return run(`vid.paused=false; vid.currentTime=${t}; scoreTick()`);},
    expire() {for (const fn of [...timers.values()]) fn();},
  };
}

for (const action of ['select', 'seek']) {
  test(`${action} releases a hung request; old completion cannot release the new owner`, async () => {
    const p = await page();
    const old = p.start(1);
    assert.equal(p.requests.length, 1);
    if (action === 'select') await p.run("selectClip({name:'b.MOV',truth:null})");
    else p.element('#vid').handlers.seeking();
    const next = p.start(2);
    assert.equal(p.requests.length, 2, 'new playback must not wait for abandoned fetch');
    assert.equal(p.requests[0].signal.aborted, true);
    await tick(); // abandoned request's catch/finally runs while new one is pending
    await p.start(4);
    assert.equal(p.requests.length, 2, 'old finally cannot release replacement owner');
    p.requests[0].resolve(response({clip:'a.MOV',t:1,knife_present:true,label:'刀/武器候选'}));
    await old;
    await tick();
    assert.equal(p.run('sticky.knife'), false);
    assert.equal(p.element('#label').textContent, '—');
    p.requests[1].resolve(response({clip:action === 'select' ? 'b.MOV':'a.MOV',t:2,
      knife_present:false,label:'未检出刀/武器'}));
    await next;
    assert.equal(p.element('#label').textContent, '未检出刀/武器');
    assert.equal(p.timers.size, 0, 'settled requests clear deadlines');
  });
}

for (const stalled of ['fetch', 'body']) {
  test(`deadline covers stalled ${stalled} and permits recovery`, async () => {
    const p = await page(), body = deferred();
    const old = p.start(1);
    if (stalled === 'body') p.requests[0].resolve({ok:true,json:() => body.promise});
    await tick();
    p.expire();
    await tick();
    assert.equal(p.element('#label').textContent, '存疑');
    assert.match(p.element('#state').textContent, /不可用/);
    assert.equal(p.requests[0].signal.aborted, true);
    const next = p.start(3);
    assert.equal(p.requests.length, 2);
    const late = {clip:'a.MOV',t:1,knife_present:true,label:'刀/武器候选'};
    if (stalled === 'body') body.resolve(late);
    else p.requests[0].resolve(response(late));
    await old;
    await tick();
    assert.equal(p.run('sticky.knife'), false);
    p.requests[1].resolve(response({clip:'a.MOV',t:3,knife_present:false,label:'未检出刀/武器'}));
    await next;
    assert.equal(p.element('#label').textContent, '未检出刀/武器');
    assert.equal(p.timers.size, 0);
  });
}

test('reselecting the same clip object cannot restore an older overlay', async () => {
  const p = await page(), requests = [];
  p.context.fetch = (url, options) => {
    const pending = deferred(); requests.push({...pending, signal:options?.signal});
    return pending.promise;
  };
  const old = p.run("CLIPS=[{name:'a.MOV',truth:null}]; selectClip(CLIPS[0])");
  const next = p.run('selectClip(CLIPS[0])');
  requests[1].resolve(response({clip:'a.MOV',frames:[],version:'new'}));
  await next;
  requests[0].resolve(response({clip:'a.MOV',frames:[],version:'old'}));
  await old;
  await tick();
  assert.equal(p.run('boxes.version'), 'new');
  assert.equal(requests[0].signal.aborted, true);
  assert.equal(p.timers.size, 0);
});

test('overlay deadline fails visibly and a later selection recovers', async () => {
  const p = await page();
  p.context.fetch = () => new Promise(() => {});
  p.run("selectClip({name:'b.MOV',truth:null})");
  p.expire();
  await tick();
  assert.match(p.element('#log').textContent, /物体框不可用/);
  p.context.fetch = async () => response({clip:'b.MOV',frames:[]});
  await p.run('selectClip(current)');
  assert.equal(p.run('boxes.clip'), 'b.MOV');
  assert.equal(p.timers.size, 0);
});

test('leaving the page cancels current work without showing stale failures', async () => {
  const p = await page();
  const pending = p.start(1);
  assert.equal(typeof p.lifecycle.pagehide, 'function');
  p.lifecycle.pagehide();
  await pending;
  assert.equal(p.requests[0].signal.aborted, true);
  assert.equal(p.timers.size, 0);
  assert.equal(p.element('#label').textContent, '—');
});
