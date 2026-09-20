/* Browser acceptance for the real local console; no model inference is claimed. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(require.resolve('playwright', {
  paths: [process.env.CONTOUR_NODE_MODULES || process.cwd()],
}));

(async () => {
  const root = path.resolve(__dirname, '..');
  const origin = process.env.CONTOUR_TEST_URL || 'http://127.0.0.1:8093';
  const key = fs.readFileSync(path.join(root, 'workers/contour/secrets/node-key'), 'utf8').trim();
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CONTOUR_BROWSER_CHANNEL ? {channel: process.env.CONTOUR_BROWSER_CHANNEL} : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    const origins = new Set();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => origins.add(new URL(request.url()).origin));
    await page.goto(origin + '/');
    await page.locator('#access-key').fill('x'.repeat(40));
    await page.locator('#connect-button').click();
    await page.waitForFunction(() => document.querySelector('#notice').textContent.includes('不正确'));
    await page.locator('#access-key').fill(key);
    await page.locator('#connect-button').click();
    await page.locator('#disconnect-button').waitFor({state: 'visible'});
    assert.equal(await page.locator('#service-state').innerText(), '等待启用');
    assert.equal(await page.locator('#gpu-state').innerText(), '未就绪');
    assert.equal(await page.locator('#model-state').innerText(), '已挂载');
    assert.equal(await page.locator('#access-key').inputValue(), '');
    assert.equal(await page.evaluate(() => localStorage.length + sessionStorage.length), 0);
    await page.locator('#admission-button').click();
    await page.waitForFunction(() => document.querySelector('#admission-button').textContent.includes('恢复'));
    await page.locator('#admission-button').click();
    await page.waitForFunction(() => document.querySelector('#admission-button').textContent.includes('暂停'));
    const content = await page.locator('body').innerText();
    assert(!/\bSAM\b|sam31|sam3\.1|facebook|huggingface/i.test(content));
    assert(!content.includes(key));
    await page.screenshot({path: path.join(root, 'tmp/contour-desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(root, 'tmp/contour-mobile.png'), fullPage: true});
    await page.locator('#disconnect-button').click();
    await page.locator('#connect-form').waitFor({state: 'visible'});
    assert.deepEqual(errors, []);
    assert.deepEqual([...origins], [origin]);
    console.log('Console passed: authentication, real status, pause/resume, branding, desktop/mobile layout, no key storage or external assets.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
