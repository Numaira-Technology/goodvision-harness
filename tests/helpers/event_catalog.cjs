// Backend contract fixture; production JavaScript contains no event inventory.
const names={fight:"打架",smoking:"吸烟",fall:"跌倒",weapon:"持械",ppe_violation:"防护装备缺失",ground_litter:"地面垃圾",stranger:"陌生人"};
const aliases={fight:["fighting"],smoking:["smoke"],fall:["falling"],weapon:["weapon_possession","holding_weapon"],stranger:["unknown_person"]};
function payload(ids=Object.keys(names)){
  return {event_catalog:{version:1,events:ids.map(id=>({id,labels:{zh:names[id]||id,en:id},aliases:aliases[id]||[],
    kind:id==="stranger"?"identity":"detection",scope:id==="ground_litter"?"scene":"person",default_enabled:true,
    ...(id==="stranger"?{}:{sensitivity:{default:id==="smoking"?"high":"standard",options:[
      {value:"standard",label:"标准敏感度",description:"必须明确识别烟具"},
      {value:"high",label:"高敏感度",description:"允许结合动作和上下文判断"}]}})
  }))}};
}
module.exports={payload};
