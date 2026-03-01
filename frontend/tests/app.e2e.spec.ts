import { expect, test } from "@playwright/test";

test("submit with Enter and restore from history", async ({ page }) => {
  let runCount = 0;
  const seenFreshModes: boolean[] = [];
  const seenThreadIds: Array<string | undefined> = [];
  const seenProfiles: Array<string | undefined> = [];

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
        thread_id: "thread-1",
        turn_index: 2,
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
        thread_id: "thread-1",
        turn_index: 1,
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

  await page.route("**/api/magi/chat", async (route) => {
    const requestBody = route.request().postDataJSON() as { fresh_mode?: boolean; thread_id?: string; profile?: string };
    seenFreshModes.push(Boolean(requestBody.fresh_mode));
    seenThreadIds.push(requestBody.thread_id);
    seenProfiles.push(requestBody.profile);
    runCount += 1;
    const payload = runCount === 1
      ? {
          run_id: "run-1",
          thread_id: "thread-1",
          turn_index: 1,
          profile: "cost",
          reply: "consensus-first",
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
          thread_id: "thread-1",
          turn_index: 2,
          profile: "cost",
          reply: "consensus-second",
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
  const runDetails = page.locator("details").filter({ hasText: "run details / routing / feedback" }).first();
  await page.getByRole("checkbox", { name: "fresh mode" }).check();
  await promptInput.fill("first prompt");
  await promptInput.press("Enter");
  await runDetails.evaluate((el) => {
    (el as HTMLDetailsElement).open = true;
  });

  await expect(page.getByText("run_id: run-1").first()).toBeVisible();
  await expect(page.getByText("prompt: first prompt")).toBeVisible();

  await promptInput.fill("second prompt");
  await promptInput.press("Enter");
  await runDetails.evaluate((el) => {
    (el as HTMLDetailsElement).open = true;
  });

  await expect(page.getByText("run_id: run-2").first()).toBeVisible();
  await expect(page.getByText("prompt: second prompt")).toBeVisible();

  await page.getByRole("button", { name: /prompt: first prompt/ }).click();
  await runDetails.evaluate((el) => {
    (el as HTMLDetailsElement).open = true;
  });

  await expect(promptInput).toHaveValue("first prompt");
  await expect(page.getByText("run_id: run-1").first()).toBeVisible();
  expect(seenFreshModes).toEqual([true, true]);
  expect(seenThreadIds).toEqual([undefined, "thread-1"]);
  expect(seenProfiles[0]).toBeUndefined();
});

test("chat-only UI does not render interaction selector", async ({ page }) => {
  await page.route("**/api/magi/profiles", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default_profile: "cost",
        profiles: ["cost", "balance", "performance", "local_only"]
      })
    });
  });
  await page.route("**/api/magi/history?limit=20&offset=0", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ total: 0, items: [] })
    });
  });

  await page.goto("/");
  await expect(page.locator('label:has-text("interaction:")')).toHaveCount(0);
});

test("local_only chat run shows conclusion elapsed time", async ({ page }) => {
  await page.route("**/api/magi/profiles", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default_profile: "cost",
        profiles: ["cost", "balance", "performance", "local_only"]
      })
    });
  });
  await page.route("**/api/magi/history?limit=20&offset=0", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ total: 0, items: [] })
    });
  });
  await page.route("**/api/magi/chat", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        run_id: "run-local-1",
        thread_id: "thread-local-1",
        turn_index: 1,
        profile: "local_only",
        reply: "やあ。",
        results: [
          {
            agent: "A",
            provider: "ollama",
            model: "qwen2.5:7b-instruct-q4_K_M",
            text: "やあ。",
            status: "OK",
            latency_ms: 45
          }
        ],
        consensus: {
          provider: "ollama",
          model: "qwen2.5:7b-instruct-q4_K_M",
          text: "やあ。",
          status: "OK",
          latency_ms: 45
        }
      })
    });
  });

  await page.goto("/");
  await page.locator('label:has-text("mode:") select').selectOption("local_only");
  const promptInput = page.getByRole("textbox", { name: "Type your prompt..." });
  const runDetails = page.locator("details").filter({ hasText: "run details / routing / feedback" }).first();
  await promptInput.fill("やあ");
  await promptInput.press("Enter");
  await runDetails.evaluate((el) => {
    (el as HTMLDetailsElement).open = true;
  });

  await expect(page.getByText(/^elapsed \d+m \d+s$/)).toBeVisible();
  await expect(runDetails.getByText("run_id: run-local-1")).toBeVisible();
});

test("settings dialog can delete all threads", async ({ page }) => {
  await page.route("**/api/magi/profiles", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default_profile: "cost",
        profiles: ["cost", "balance", "performance", "local_only"]
      })
    });
  });
  await page.route("**/api/magi/history?limit=20&offset=0", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        total: 2,
        items: [
          {
            run_id: "run-2",
            thread_id: "thread-2",
            turn_index: 1,
            profile: "cost",
            prompt: "thread2 prompt",
            created_at: "2026-02-10T00:00:00+00:00",
            results: [
              { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A2", status: "OK", latency_ms: 80 },
              { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B2", status: "OK", latency_ms: 90 },
              { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C2", status: "OK", latency_ms: 100 }
            ],
            consensus: {
              provider: "openai",
              model: "gpt-4.1-mini",
              text: "consensus-2",
              status: "OK",
              latency_ms: 70
            }
          },
          {
            run_id: "run-1",
            thread_id: "thread-1",
            turn_index: 1,
            profile: "cost",
            prompt: "thread1 prompt",
            created_at: "2026-02-09T00:00:00+00:00",
            results: [
              { agent: "A", provider: "openai", model: "gpt-4.1-mini", text: "A1", status: "OK", latency_ms: 80 },
              { agent: "B", provider: "anthropic", model: "claude-sonnet-4-20250514", text: "B1", status: "OK", latency_ms: 90 },
              { agent: "C", provider: "gemini", model: "gemini-2.5-flash", text: "C1", status: "OK", latency_ms: 100 }
            ],
            consensus: {
              provider: "openai",
              model: "gpt-4.1-mini",
              text: "consensus-1",
              status: "OK",
              latency_ms: 70
            }
          }
        ]
      })
    });
  });

  let deleteCalled = false;
  await page.route("**/api/magi/history", async (route) => {
    if (route.request().method() !== "DELETE") {
      await route.continue();
      return;
    }
    deleteCalled = true;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ deleted_runs: 2, deleted_threads: 2 })
    });
  });

  await page.goto("/");
  await expect(page.getByText("thread1 prompt")).toBeVisible();

  await page.getByRole("button", { name: "Open settings" }).click();
  page.once("dialog", async (dialog) => {
    await dialog.accept();
  });
  await page.getByRole("button", { name: "Delete All Threads" }).click();

  expect(deleteCalled).toBe(true);
  await expect(page.getByText("No history yet.")).toBeVisible();
});
