const assert=require("node:assert/strict");
const vm=require("node:vm");
const {test}=require("node:test");
const {createDocument, runDashboard} = require("./helpers/dashboard.cjs");

function dashboard(){
  const {document, element} = createDocument();
  const requests=[];
  const context=vm.createContext({
    document,
    window:{isSecureContext:true},location:{hostname:"localhost"},
    fetch:async(url,options)=>{
      requests.push({url,options});
      if(context.fail)return {ok:false,json:async()=>({detail:"save failed"})};
      const id=JSON.parse(options.body).route_id;
      return {ok:true,json:async()=>({...context.app.state.routes,selected_id:id,
        selected_label:"线路 "+id,selection_method:"manual",selection_persisted:true})}
    },
  });
  runDashboard(context, `
    state.routes={supported:true,can_select:true,can_probe:true,state:"idle",
      selected_id:"A",selected_label:"线路 A",trials_per_route:3,
      results:["A","B"].map(id=>({id,label:"线路 "+id,state:"pending",samples:[],median_seconds:null}))};
    globalThis.app={state,renderRoutes,selectRoute,controls};renderRoutes();
  `);
  return {element,requests,context,...context.app}
}

test("manual route buttons submit public IDs and show persisted selection",async()=>{
  const view=dashboard();
  let buttons=view.element("route-results").querySelectorAll("button");
  assert.equal(buttons[0].disabled,true);
  assert.equal(buttons[1].disabled,false);
  await buttons[1].onclick();
  assert.equal(view.requests[0].url,"/v1/deployment/routes/select");
  assert.deepEqual(JSON.parse(view.requests[0].options.body),{route_id:"B"});
  assert.match(view.element("route-selected").textContent,/线路 B.*已保存/);
  assert.match(view.element("route-note").textContent,/已手动选择线路 B/);
  buttons=view.element("route-results").querySelectorAll("button");
  assert.equal(buttons[1].disabled,true)
});

test("manual selection is disabled during detection and route probes",async()=>{
  const view=dashboard();
  view.state.running=true;view.controls();
  assert.ok(view.element("route-results").querySelectorAll("button").every(b=>b.disabled));
  await view.selectRoute("B");assert.equal(view.requests.length,0);
  view.state.running=false;view.state.routes.state="running";view.state.routes.can_select=false;
  view.controls();
  assert.ok(view.element("route-results").querySelectorAll("button").every(b=>b.disabled));
  await view.selectRoute("B");assert.equal(view.requests.length,0)
});

test("failed persistence leaves the selected route unchanged",async()=>{
  const view=dashboard();view.context.fail=true;
  await view.selectRoute("B");
  assert.equal(view.state.routes.selected_id,"A");
  assert.match(view.element("route-note").textContent,/未保存，保留原选择/);
  assert.equal(view.state.routeSubmitting,false)
});
