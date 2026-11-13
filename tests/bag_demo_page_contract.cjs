// Exercise the actual inline page code with delayed responses, without a browser/server.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('tools/bag_demo/page.html', 'utf8');
const code = html.match(/<script>([\s\S]*)<\/script>/)[1];
const nodes = new Map();
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {
    textContent: '', innerHTML: '', style: {}, dataset: {}, handlers: {},
    classList: {toggle() {}}, appendChild() {}, load() {},
    paused: true, ended: false, currentTime: 1, checked: false,
    addEventListener(name, fn) {this.handlers[name] = fn;},
    getContext() {return {};},
  });
  return nodes.get(id);
}
const response = data => ({ok: true, json: async () => data});
const context = vm.createContext({console, AbortController, setTimeout, clearTimeout,
  window: {addEventListener() {}}, requestAnimationFrame() {},
  document: {querySelector: element, querySelectorAll: () => [], createElement: element},
  fetch: async () => response({clips: []}),
});
vm.runInContext(code, context);
const run = text => vm.runInContext(text, context);
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  await tick();
  let finishBoxes;
  context.fetch = url => url.endsWith('a.MOV')
    ? new Promise(resolve => {finishBoxes = resolve;})
    : Promise.resolve(response({clip: 'b.MOV', frames: []}));
  const oldSelect = run("selectClip({name:'a.MOV',truth:null})");
  await run("selectClip({name:'b.MOV',truth:null})");
  finishBoxes(response({clip: 'a.MOV', frames: []}));
  await oldSelect;
  assert.equal(run('boxes.clip'), 'b.MOV', 'old overlay cannot replace current clip');

  for (const action of ['select', 'seek']) {
    let finishScore;
    context.fetch = async () => response({frames: []});
    await run("selectClip({name:'a.MOV',truth:null})");
    context.fetch = url => url.startsWith('/api/verdict')
      ? new Promise(resolve => {finishScore = resolve;})
      : Promise.resolve(response({frames: []}));
    run('vid.paused=false; vid.currentTime=1;');
    const scoring = run('scoreTick()');
    if (action === 'select') await run("selectClip({name:'b.MOV',truth:null})");
    else element('#vid').handlers.seeking();
    finishScore(response({clip:'a.MOV',t:1,knife_present:true,knife:true,
      knife_p:.99,label:'刀/武器候选'}));
    await scoring;
    assert.equal(run('sticky.knife'), false, 'stale verdict must not latch after ' + action);
    assert.equal(element('#label').textContent, '—');
    assert.equal(run('scoreRequest'), null);
  }

  context.fetch = async () => response({frames: []});
  await run("selectClip({name:'b.MOV',truth:{knife:true,alcohol:false}})");
  run('vid.paused=false; vid.currentTime=1;');
  context.fetch = async () => response({clip:'b.MOV',t:1,knife_present:false,
    knife:true,knife_p:.88,weapon_native:false,label:'存疑'});
  await run('scoreTick()');
  assert.equal(element('#label').textContent, '存疑', 'below-gate positive is uncertain');
  element('#chkTruth').checked = true;
  element('#chkTruth').handlers.change();
  assert.match(element('#selfcheck').textContent, /非本窗口真值/);
  element('#chkTruth').checked = false;
  element('#chkTruth').handlers.change();
  assert.equal(element('#selfcheck').textContent, '');

  context.fetch = async () => ({ok:false, json:async () => ({error:'unavailable'})});
  run('vid.currentTime=3;');
  await run('scoreTick()');
  assert.equal(element('#label').textContent, '存疑');
  assert.equal(element('#knifeP').textContent, '–');
  assert.match(element('#state').textContent, /不可用/);
  console.log('PASS: stale overlays, clip/seek verdict races, uncertainty, errors and truth toggle');
})().catch(error => {console.error(error); process.exitCode = 1;});
