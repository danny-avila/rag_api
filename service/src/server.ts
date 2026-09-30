import { createApp } from "./app";
import { fromEnv } from "./config";

if (import.meta.main) {
  try {
    const config = fromEnv(process.env);
    const port = Number(process.env.RAG_PORT ?? 8001);
    if (!Number.isInteger(port) || port < 1 || port > 65535)
      throw new Error("Invalid port");
    const server = Bun.serve({
      hostname: process.env.RAG_HOST ?? "0.0.0.0",
      port,
      maxRequestBodySize: config.maxBodyBytes,
      fetch: createApp(config).fetch,
    });
    // Graceful shutdown lets active requests finish; their deadlines stay bounded.
    for (const event of ["SIGINT", "SIGTERM"] as const) {
      process.once(event, () => {
        void server.stop(false);
      });
    }
    console.info(`RAG Bun service listening on port ${server.port}`);
  } catch {
    console.error(
      "Unable to start RAG Bun service: check runtime configuration",
    );
    process.exitCode = 1;
  }
}
