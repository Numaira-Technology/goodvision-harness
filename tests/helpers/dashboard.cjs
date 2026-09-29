/* Shared DOM and script loading only. Each suite owns its clocks, requests,
 * and initial state so tests continue to execute the shipped dashboard logic.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {payload} = require("./event_catalog.cjs");
const catalogScript = new vm.Script(fs.readFileSync(path.join(__dirname,"../../src/goodvision_harness/assets/event_catalog.js"),"utf8"));

const filename = path.join(__dirname, "../../src/goodvision_harness/dashboard.html");
const html = fs.readFileSync(filename, "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "Dashboard must contain its application script");
new vm.Script(script, {filename}); // Check browser startup syntax as well.
const startup = script.indexOf("ui.defaultStreamUrl.oninput=");
assert.ok(startup > 0, "Dashboard browser startup boundary must exist");
const handlers = new vm.Script(script.slice(0, startup), {filename});

class Element {
  constructor(tag = "div", rejectInnerHTML = false) {
    this.tag = tag; this.textContent = ""; this.className = "";
    this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false;
    if (rejectInnerHTML) {
      Object.defineProperty(this, "innerHTML", {set(value) {
        throw new Error("Untrusted content must not reach innerHTML: " + value);
      }});
    }
    this.classList = {
      add: name => { if (!this.className.split(" ").includes(name)) this.className += " " + name; },
      remove: name => { this.className = this.className.split(" ").filter(item => item !== name).join(" "); },
      toggle: (name, active) => active ? this.classList.add(name) : this.classList.remove(name),
    };
  }
  append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
  appendChild(item) { this.append(item); }
  prepend(item) { item.parent = this; this.children.unshift(item); }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  remove() { this.parent.children = this.parent.children.filter(item => item !== this); }
  get lastChild() { return this.children.at(-1); }
  querySelectorAll(tag) {
    return this.children.flatMap(child => [
      ...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag),
    ]);
  }
  getAttribute(name) { return this[name] || null; }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  scrollIntoView() {}
  focus() { this.focused = true; }
}

function createDocument(features = [], {rejectInnerHTML = false} = {}) {
  const elements = new Map();
  const createElement = tag => new Element(tag, rejectInnerHTML);
  const inputs = features.map(feature => Object.assign(createElement("input"), {
    checked: true, dataset: {feature},
  }));
  const element = id => {
    if (!elements.has(id)) elements.set(id, createElement("div"));
    return elements.get(id);
  };
  return {element, document: {
    testFeatures: features,
    hidden: false, getElementById: element, createElement,
    querySelectorAll: selector => selector === "[data-event-presets]" ? [element("test-presets")] : inputs,
  }};
}

function runDashboard(context, setup) {
  catalogScript.runInContext(context);
  handlers.runInContext(context);
  context.testCatalog = payload();
  vm.runInContext(`renderEventCatalog(testCatalog);
    for(const input of featureInputs)input.checked=document.testFeatures.includes(input.dataset.feature);`, context);
  vm.runInContext(setup, context);
}

module.exports = {createDocument, runDashboard, html, script};
