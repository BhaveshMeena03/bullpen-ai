// Render demo/og-card.html to demo/og-image.png (1200x630 social preview).
//   node scripts/make_og_image.mjs
import { chromium } from "playwright-core";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.goto("file://" + join(root, "demo", "og-card.html"));
await page.waitForTimeout(400);
await page.screenshot({ path: join(root, "demo", "og-image.png") });
await browser.close();
console.log("wrote demo/og-image.png");
