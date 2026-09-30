const assert=require('node:assert/strict');
require('../src/goodvision_harness/assets/event_catalog.js').accept(require('./helpers/event_catalog.cjs').payload());
const {test}=require('node:test');
const {translate}=require('../src/goodvision_harness/assets/console_i18n.js');
const {drawSubjectLabel,eventObservationSummary}=require('../src/goodvision_harness/camera_display.js');

test('English covers navigation, sensitivity explanations, placeholders and errors',()=>{
  assert.equal(translate('实时工作台','en'),'Live workspace');
  assert.equal(translate('部署与设置','en'),'Deployment & settings');
  assert.equal(translate('必须明确识别烟具','en'),'Requires clearly visible smoking items');
  assert.equal(translate('暂无分析结果','en'),'No analysis results yet');
  assert.equal(translate('请填写服务器提供的服务标识。','en'),'Enter the service identifier supplied by the server.');
});

test('dynamic counts and event observations retain values and confirmation distinctions',()=>{
  assert.match(translate('已部署 12 路通道，当前可用 8 路。','en'),/Deployed channels: 12; available: 8\./);
  const observation=eventObservationSummary({state:'completed',judgments:[
    {event_label:'fight',observed_detected:true,detected:null,localization_state:'unavailable',confirmation_state:'pending'},
    {event_label:'smoking',observed_detected:false,detected:true},
    {event_label:'fall',observed_detected:null,detected:false},
  ]});
  const english=translate(observation,'en');
  assert.match(english,/Fight.*Observed, pending confirmation, location pending/);
  assert.match(english,/Smoking.*Not observed/);assert.match(english,/Fall.*Pending/);
  assert.doesNotMatch(english,/[\u4e00-\u9fff]/);
});

test('legacy English copy switches to Chinese and dialog fragments retain whitespace',()=>{
  assert.equal(translate('Fullscreen','zh'),'全屏');
  assert.equal(translate('Delete ','zh')+'李明'+translate(' and all enrolled samples?','zh'),'删除 李明 及其全部人脸样本？');
  assert.equal(translate('  实时工作台\n','en'),'  Live workspace\n');
  assert.equal(translate('Waiting for the first analysis result...','zh'),'等待首个分析结果…');
});

test('unknown values and endpoint URLs are untouched',()=>{
  const endpoint='http://127.0.0.1:7900/v1/results/latest?label=打架';
  assert.equal(translate(endpoint,'en'),endpoint);
  assert.equal(translate('API / SSE / GVD1','en'),'API / SSE / GVD1');
  assert.equal(translate(null,'en'),null);assert.equal(translate(12,'en'),12);
});

test('canvas event labels switch language while person names and event data remain original',()=>{
  const lines=[],context={save(){},restore(){},fillRect(){},measureText:text=>({width:text.length*7}),fillText:text=>lines.push(text)};
  const subject={track_id:'7',display_name:'打架 · 李明',tracking_confirmation:'confirmed',alert:true,
    event_labels:['fight','smoking'],polygons:[[[.1,.3],[.5,.3],[.5,.7]]]};
  const before=JSON.stringify(subject),previous=globalThis.GoodvisionI18n;
  try{
    globalThis.GoodvisionI18n={text:value=>translate(value,'en')};
    drawSubjectLabel(context,subject,1000,600);
    assert.deepEqual(lines,['打架 · 李明','Fight · Smoking']);
    lines.length=0;globalThis.GoodvisionI18n={text:value=>translate(value,'zh')};
    drawSubjectLabel(context,subject,1000,600);
    assert.deepEqual(lines,['打架 · 李明','打架 · 吸烟']);
    assert.equal(JSON.stringify(subject),before);
  }finally{
    if(previous===undefined)delete globalThis.GoodvisionI18n;else globalThis.GoodvisionI18n=previous;
  }
});


test('new safety events retain their labels and translate in the modular console',()=>{
  const observation=eventObservationSummary({state:'completed',judgments:[
    {event_label:'ppe_violation',observed_detected:true,detected:null},
    {event_label:'ground_litter',observed_detected:true,detected:null},
  ]});
  const english=translate(observation,'en');
  assert.match(english,/Missing protective equipment/);
  assert.match(english,/Ground litter/);
  assert.doesNotMatch(english,/[\u4e00-\u9fff]/);
  assert.equal(translate('当前场景','en'),'Current scene');
});
