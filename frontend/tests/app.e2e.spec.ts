import { expect, test } from "@playwright/test";

test("submit with Enter and restore from history", async ({ page }) => {
  let runCount = 0;
  const seenFreshModes: boolean[] = [];

  await page.route("**/api/magi/profiles", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default_profile: "cost",
        profiles: ["cost", "balance", "performance"]
      })
    });
  });

  await page.route("**/api/magi/history?limit=20&offset=0", async (route) => {
    const items = [];
    if (runCount >= 2) {
      items.push({
        run_id: "run-2",
        profile: "cost",
        prompt: "second prompt",
        created_at: "2026-02-10T00:00:00+00:00",
        results: [
          { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A-second", status: "OK", latency_ms: 130 },
          { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B-second", status: "OK", latency_ms: 140 },
          { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C-second", status: "OK", latency_ms: 150 }
        ],
        consensus: {
          provider: "openai",
          model: "gpt-4.1-mini",
          text: "consensus-second",
          status: "OK",
          latency_ms: 95
        }
      });
    }
    if (runCount >= 1) {
      items.push({
        run_id: "run-1",
        profile: "cost",
        prompt: "first prompt",
        created_at: "2026-02-09T00:00:00+00:00",
        results: [
          { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A-first", status: "OK", latency_ms: 100 },
          { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B-first", status: "OK", latency_ms: 110 },
          { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C-first", status: "OK", latency_ms: 120 }
        ],
        consensus: {
          provider: "openai",
          model: "gpt-4.1-mini",
          text: "consensus-first",
          status: "OK",
          latency_ms: 90
        }
      });
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ total: items.length, items })
    });
  });

  await page.route("**/api/magi/run", async (route) => {
    const requestBody = route.request().postDataJSON() as { fresh_mode?: boolean };
    seenFreshModes.push(Boolean(requestBody.fresh_mode));
    runCount += 1;
    const payload = runCount === 1
      ? {
          run_id: "run-1",
          profile: "cost",
          results: [
            { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A-first", status: "OK", latency_ms: 100 },
            { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B-first", status: "OK", latency_ms: 110 },
            { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C-first", status: "OK", latency_ms: 120 }
          ],
          consensus: {
            provider: "openai",
            model: "gpt-4.1-mini",
            text: "consensus-first",
            status: "OK",
            latency_ms: 90
          }
        }
      : {
          run_id: "run-2",
          profile: "cost",
          results: [
            { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A-second", status: "OK", latency_ms: 130 },
            { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B-second", status: "OK", latency_ms: 140 },
            { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C-second", status: "OK", latency_ms: 150 }
          ],
          consensus: {
            provider: "openai",
            model: "gpt-4.1-mini",
            text: "consensus-second",
            status: "OK",
            latency_ms: 95
          }
        };

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload)
    });
  });

  await page.goto("/");

  const promptInput = page.getByRole("textbox", { name: "Type your prompt..." });
  await page.getByRole("checkbox", { name: "fresh mode" }).check();
  await promptInput.fill("first prompt");
  await promptInput.press("Enter");

  await expect(page.locator("span", { hasText: "run_id: run-1" })).toBeVisible();
  await expect(page.getByText("prompt: first prompt")).toBeVisible();

  await promptInput.fill("second prompt");
  await promptInput.press("Enter");

  await expect(page.locator("span", { hasText: "run_id: run-2" })).toBeVisible();
  await expect(page.getByText("prompt: second prompt")).toBeVisible();

  await page.getByRole("button", { name: /prompt: first prompt/ }).click();

  await expect(promptInput).toHaveValue("first prompt");
  await expect(page.locator("span", { hasText: "run_id: run-1" })).toBeVisible();
  expect(seenFreshModes).toEqual([true, true]);
});
