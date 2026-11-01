const {test}=require('node:test');
const assert=require('node:assert/strict');
const {SharedController,eventObservationSummary}=require('../src/goodvision_harness/camera_display.js');

test('event observations survive missing localization and keep confirmation separate',()=>{
  const summary=eventObservationSummary({state:'completed',judgments:[
    {event_label:'fight',observed_detected:true,detected:null,localization_state:'unavailable',confirmation_state:'pending'},
    {event_label:'smoking',observed_detected:false,detected:true},
    {event_label:'fall',observed_detected:null,detected:false},
    {event_label:'weapon',observed_detected:true,detected:true,confirmation_state:'confirmed'},
  ]});
  assert.match(summary,/打架：发现，待确认，定位待确认/);
  assert.match(summary,/吸烟：未发现/);assert.match(summary,/跌倒：待判断/);assert.match(summary,/持械：已确认/);
});

test('an invalid or empty VLM response never displays an all-clear',()=>{
  assert.match(eventObservationSummary({state:'failed',judgments:[]}),/等待/);
  assert.match(eventObservationSummary({state:'completed',judgments:[]}),/尚无可用/);
});

test('observation-only reports leave confirmation to the tracked scene',()=>{
  const summary=eventObservationSummary({type:'vlm_report',state:'completed',judgments:[
    {event_label:'smoking',observed_detected:true,detected:null,localization_state:'available',confirmation_state:'pending'},
    {event_label:'fight',observed_detected:true,detected:null,localization_state:'unavailable',confirmation_state:'pending'},
  ]});
  assert.equal(summary,'打架：发现，定位待确认 · 吸烟：发现');
});

test('diagnostic mirror draws only new source frames and owns no processing resources',()=>{
  let callback,cancelled=0,draws=0;
  const media={width:1280,height:730,dataset:{frameSequence:'9',mediaTimestamp:'0.6'}};
  const canvas={width:0,height:0,dataset:{},getContext:()=>({drawImage:value=>{assert.equal(value,media);draws++;}})};
  let stats={display_instance_id:'first',distinct_presented_fps:15};
  const mirror=new SharedController({canvas,source:()=>({media,stats}),
    requestFrame:fn=>{callback=fn;return 7;},cancelFrame:id=>{assert.equal(id,7);cancelled++;}});
  mirror.start();mirror.start();assert.equal(draws,1);callback();assert.equal(draws,1);
  media.dataset.frameSequence='10';callback();assert.equal(draws,2);
  assert.equal(canvas.width,1280);assert.equal(canvas.dataset.mediaTimestamp,'0.6');
  stats={...stats,display_instance_id:'reconnected'};callback();assert.equal(draws,3);
  mirror.stop();assert.equal(cancelled,1);callback();assert.equal(draws,3);
  assert.equal(mirror.controller,undefined);
});

test('diagnostic mirror waits safely for the parent canvas after reconnect',()=>{
  let callback,source=null,draws=0;
  const canvas={dataset:{},getContext:()=>({drawImage:()=>draws++})};
  const mirror=new SharedController({canvas,source:()=>source,requestFrame:fn=>{callback=fn;return 1;},cancelFrame:()=>{}});
  mirror.start();assert.equal(draws,0);
  source={media:{width:10,height:10,dataset:{frameSequence:'1'}}};callback();assert.equal(draws,1);
  source=null;callback();assert.equal(draws,1);mirror.stop();
});
require('../src/goodvision_harness/assets/event_catalog.js').accept(require('./helpers/event_catalog.cjs').payload());
