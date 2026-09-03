// Records the explainer: what happens between a question and an answer.
//
//   node scripts/make_tech_demo.mjs
//   ffmpeg -framerate 24 -i /tmp/mb_tech/f%05d.png \
//          -c:v h264_videotoolbox -b:v 8M -pix_fmt yuv420p \
//          ~/Desktop/marketbubble-how-it-works.mp4
//
// The companion to make_platform_demo.mjs. That one shows the product
// working; this one shows why it works, for the reading of "how it works"
// that means the mechanism rather than the demo.
//
// Reveals one stage at a time and holds on each, so it can be read at
// normal speed without pausing. The page is demo/how-it-works.html, and
// every number in it comes from the running system rather than being
// rounded for effect — when the archive grows, the page is the one place
// to correct.

import { chromium } from "playwright-core";
import { mkdirSync, rmSync } from "fs";
import { resolve } from "path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FRAMES = "/tmp/mb_tech";
// ?staged suppresses the page's own reveal-everything fallback, which
// exists so the page is readable when somebody just opens it.
const PAGE = "file://" + resolve("demo/how-it-works.html") + "?staged";

rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1.5,          // renders at exactly 1920x1080
});
await page.goto(PAGE, { waitUntil: "load" });
await page.waitForTimeout(500);

let n = 0;
const shoot = async (count = 1) => {
  for (let i = 0; i < count; i++) {
    await page.screenshot({
      path: `${FRAMES}/f${String(n++).padStart(5, "0")}.png`,
    });
  }
};

// Reveal a step, hold on it, then let it settle back so the next one is
// what the eye goes to. `lit` is the brighter border during its own beat.
const reveal = async (id, hold) => {
  await page.evaluate((sel) => {
    document.querySelectorAll(".step.lit")
      .forEach((el) => el.classList.remove("lit"));
    const el = document.getElementById(sel);
    el.classList.add("on", "lit");
  }, id);
  // The CSS transition is 350ms; shoot through it so the fade is on film.
  for (let i = 0; i < 9; i++) {
    await page.waitForTimeout(45);
    await shoot(1);
  }
  await shoot(hold);
};

await shoot(26);                                   // title alone

await reveal("s1", 40);   // transcribe
await reveal("s2", 34);   // windows
await reveal("s3", 42);   // embeddings
await reveal("s4", 46);   // exact terms — the part worth dwelling on
await reveal("s5", 38);   // rerank
await reveal("s6", 44);   // the answer

await page.evaluate(() => {
  document.querySelectorAll(".step.lit")
    .forEach((el) => el.classList.remove("lit"));
  document.getElementById("rule").classList.add("on");
});
for (let i = 0; i < 10; i++) {
  await page.waitForTimeout(45);
  await shoot(1);
}
await shoot(60);                                   // hold on the promise

await browser.close();
console.log(`captured ${n} frames -> ${FRAMES}`);
console.log(
  "ffmpeg -framerate 24 -i /tmp/mb_tech/f%05d.png " +
  "-c:v h264_videotoolbox -b:v 8M -pix_fmt yuv420p " +
  "~/Desktop/marketbubble-how-it-works.mp4");
