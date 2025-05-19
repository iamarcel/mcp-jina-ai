import express, { Request, Response } from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { z } from "zod";
import fetch from "node-fetch"; // Use node-fetch for CJS compatibility or ensure Node >= 18 for global fetch
import {
  SearchResponseSchema,
  ReaderResponseSchema,
  GroundingResponseSchema,
  EmbeddingResponseSchema,
  SearchResponse,
  ReaderResponse,
  GroundingResponse,
  EmbeddingResponse,
} from "./schemas.js";

// --- Zod Schemas ---

// Schema for the standard MCP tool output format
const McpContentSchema = z.object({
  content: z
    .array(
      z.object({
        type: z.literal("text"),
        text: z.string(),
      })
    )
    .min(1, "Content array cannot be empty"),
});

// Schema for the Search tool input
const SearchInputSchema = {
  query: z.string().describe("The search query to search the web for."),
  // Optional: site: z.string().url().optional().describe("Limit search to this specific website URL.")
};

// Schema for the Fact Check tool input
const FactCheckInputSchema = {
  statement: z
    .string()
    .describe("The statement to verify for factual accuracy."),
  // Optional: site: z.string().optional().describe("Comma-separated list of URLs to use as grounding references.")
};

// Schema for the Read Webpage tool input
const ReadWebpageInputSchema = {
  url: z.string().describe("The URL of the webpage to read."), // `.url()` is not supported by Gemini
  // Optional: returnFormat: z.enum(["markdown", "html", "text", "screenshot", "pageshot"]).optional().default("text").describe("Desired format of the returned content."),
  query: z
    .string()
    .describe(
      "Query used to select the most relevant parts of the webpage content."
    ),
}; 

// --- Jina API Configuration ---

// Get your Jina AI API key for free: https://jina.ai/?sui=apikey
const JINA_API_KEY = process.env.JINA_API_KEY;
const JINA_SEARCH_URL = "https://s.jina.ai/";
const JINA_GROUNDING_URL = "https://g.jina.ai/";
const JINA_READER_URL = "https://r.jina.ai/";
const JINA_EMBEDDING_URL = "https://r.jina.ai/v1/embeddings";

if (!JINA_API_KEY) {
  console.error("Error: JINA_API_KEY environment variable is not set.");
  console.log(
    "Please get your Jina AI API key for free: https://jina.ai/?sui=apikey and set it as an environment variable."
  );
  process.exit(1); // Exit if the key is not found
}

const JINA_HEADERS = {
  Authorization: `Bearer ${JINA_API_KEY}`,
  Accept: "application/json",
  "Content-Type": "application/json",
};

// --- Embedding and Similarity Helpers ---

/** Split text into manageable chunks for embedding generation. */
function chunkText(text: string, chunkSize = 200): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

// Generate embeddings using Jina AI with basic validation
async function embedTexts(texts: string[]): Promise<number[][]> {
  const trimmed = texts.map((t) => t.trim()).filter((t) => t !== "");
  if (trimmed.length === 0) {
    return [];
  }
  const response = await embedJina(trimmed);
  if ("data" in response) {
    return response.data.map((d) => d.embedding);
  }
  return response.embeddings;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length && i < b.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// --- Helper Function for Jina API Calls ---

async function callJinaApi<T>(
  url: string,
  body: object,
  headers: Record<string, string> = JINA_HEADERS
): Promise<T> {
  // Basic retry mechanism (e.g., 1 retry on network failure)
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        let errorBody = "Could not parse error response.";
        try {
          errorBody = await response.text();
        } catch (parseError) {
          // Ignore if parsing fails
        }
        throw new Error(
          `Jina API request failed at ${url}: ${response.status} ${response.statusText} - ${errorBody}`
        );
      }

      // Check if response body is empty before parsing JSON
      const text = await response.text();
      if (!text) {
        throw new Error(
          `Jina API request succeeded at ${url} but returned an empty response.`
        );
      }
      return JSON.parse(text) as T; // Parse JSON only if text is not empty
    } catch (error: any) {
      console.error(`Attempt ${attempt} failed for ${url}:`, error.message);
      if (attempt === 2) {
        // If it's the last attempt, rethrow the error
        throw error;
      }
      // Optional: Add a small delay before retrying
      // await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
  // Should not be reached if retry logic is correct, but satisfies TypeScript
  throw new Error(
    `Jina API request failed after multiple attempts for ${url}.`
  );
}

// Specialized helpers for each Jina endpoint with typed responses
async function searchJina(query: string): Promise<SearchResponse> {
  const raw = await callJinaApi<SearchResponse>(JINA_SEARCH_URL, { q: query });
  return SearchResponseSchema.parse(raw);
}

async function readJina(url: string): Promise<ReaderResponse> {
  const raw = await callJinaApi<ReaderResponse>(JINA_READER_URL, { url });
  return ReaderResponseSchema.parse(raw);
}

async function groundJina(statement: string): Promise<GroundingResponse> {
  const raw = await callJinaApi<GroundingResponse>(JINA_GROUNDING_URL, {
    statement,
  });
  return GroundingResponseSchema.parse(raw);
}

async function embedJina(texts: string[]): Promise<EmbeddingResponse> {
  const raw = await callJinaApi<EmbeddingResponse>(JINA_EMBEDDING_URL, {
    input: texts,
    model: "jina-embeddings-v3",
    task: "text-matching",
    late_chunking: true,
  });
  return EmbeddingResponseSchema.parse(raw);
}

// --- MCP Server Setup ---

const server = new McpServer(
  {
    name: "jina-search-tools-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// --- Define Tools ---

// 1. Search Tool (using s.jina.ai)
server.tool(
  "search",
  "Search the web for information, for example recent information.",
  SearchInputSchema,
  async ({ query }): Promise<z.infer<typeof McpContentSchema>> => {
    console.log(`Executing search tool with query: "${query}"`);
    try {
      const response = await searchJina(query);

      // Ensure response.data is an array before proceeding
      if (response.data.length === 0) {
        return {
          content: [{ type: "text", text: "No search results found." }],
        };
      }


      // Embed the query once
      const queryEmbeddingArray = await embedTexts([query.trim()]);
      const queryEmbedding = queryEmbeddingArray[0];

      // Process each result by selecting the most relevant chunks
      const processed = await Promise.all(
        response.data.map(async (item: SearchResponse["data"][number], index: number) => {
          const allChunks = chunkText(item.content || "");
          const validChunks = allChunks.filter((c) => c.trim() !== "");
          if (validChunks.length === 0) {
            return `Result ${index + 1}:\nTitle: ${item.title}\nURL: ${item.url}\nRelevant Content: \n\n`;
          }
          const chunkEmbeddings = await embedTexts(validChunks);
          if (chunkEmbeddings.length !== validChunks.length) {
            console.error(
              `Embedding count mismatch for search result ${item.url}. Expected ${validChunks.length}, got ${chunkEmbeddings.length}.`
            );
          }
          const scored = validChunks
            .map((c, i) => ({
              c,
              score:
                queryEmbedding && chunkEmbeddings[i]
                  ? cosineSimilarity(queryEmbedding, chunkEmbeddings[i])
                  : 0,
            }))
            .sort((a, b) => b.score - a.score);
          const bestChunks = scored.slice(0, 5).map((s) => s.c).join("\n\n");
          return `Result ${index + 1}:\nTitle: ${item.title}\nURL: ${item.url}\nRelevant Content:\n${bestChunks}\n---`;
        })
      );

      const combinedContent = processed.join("\n\n");

      // Validate output before returning
      return McpContentSchema.parse({
        content: [{ type: "text", text: combinedContent }],
      });
    } catch (error: any) {
      console.error("Error executing search tool:", error);
      // Return error message in the expected format
      return {
        content: [{ type: "text", text: `Search failed: ${error.message}` }],
      };
    }
  }
);

// 2. Fact Check Tool (using g.jina.ai)
server.tool(
  "fact-check",
  "Verify the accuracy of a statement by checking it against reliable sources.",
  FactCheckInputSchema,
  async ({ statement }): Promise<z.infer<typeof McpContentSchema>> => {
    console.log(`Executing fact-check tool with statement: "${statement}"`);
    try {
      const response = await groundJina(statement);

      const { result, reason, references } = response.data;
      let referencesText = "No specific references provided.";
      if (Array.isArray(references) && references.length > 0) {
        referencesText = references
          .map(
            (ref: any, index: number) =>
              `Reference ${index + 1}:\n  URL: ${ref.url}\n  Quote: "${
                ref.keyQuote
              }"\n  Supportive: ${ref.isSupportive}`
          )
          .join("\n");
      }

      const outputText = `Statement: "${statement}"\nResult: ${
        result ? "Likely True" : "Likely False"
      }\nReason: ${
        reason || "No reason provided."
      }\n\nReferences:\n${referencesText}`;

      // Validate output before returning
      return McpContentSchema.parse({
        content: [{ type: "text", text: outputText }],
      });
    } catch (error: any) {
      console.error("Error executing fact-check tool:", error);
      // Return error message in the expected format
      return {
        content: [
          { type: "text", text: `Fact check failed: ${error.message}` },
        ],
      };
    }
  }
);

// 3. Read Webpage Tool (using r.jina.ai)
server.tool(
  "read-webpage",
  "Read a webpage and extract its content.",
  ReadWebpageInputSchema,
  async ({ url, query }): Promise<z.infer<typeof McpContentSchema>> => {
    console.log(`Executing read-webpage tool for URL: ${url}`);
    try {
      // Add X-Return-Format header if needed, default is markdown-like text
      // const headers = { ...JINA_HEADERS, 'X-Return-Format': 'text' };
      const response = await readJina(url);

      const { title, content } = response.data;
      const allChunks = chunkText(content || "");
      const validChunks = allChunks.filter((c) => c.trim() !== "");
      let topChunks = "";
      if (validChunks.length > 0) {
        const chunkEmbeddings = await embedTexts(validChunks);
        let queryText = query.trim();
        if (queryText === "") {
          queryText = title || "";
        }
        const queryEmbeddingArray = await embedTexts([queryText]);
        const queryEmbedding = queryEmbeddingArray[0];
        if (queryEmbedding && chunkEmbeddings.length === validChunks.length) {
          const scored = validChunks
            .map((c, i) => ({
              c,
              score: cosineSimilarity(queryEmbedding, chunkEmbeddings[i]),
            }))
            .sort((a, b) => b.score - a.score);
          topChunks = scored.slice(0, 5).map((s) => s.c).join("\n\n");
        } else {
          topChunks = validChunks.slice(0, 5).join("\n\n");
        }
      }

      const outputText = `Title: ${title || "N/A"}\nURL: ${url}\n\nRelevant Content:\n${
        topChunks || "No content extracted."
      }`;

      // Validate output before returning
      return McpContentSchema.parse({
        content: [{ type: "text", text: outputText }],
      });
    } catch (error: any) {
      console.error("Error executing read-webpage tool:", error);
      // Return error message in the expected format
      return {
        content: [
          { type: "text", text: `Reading webpage failed: ${error.message}` },
        ],
      };
    }
  }
);

// --- Express SSE Server Setup ---

const app = express();
const port = process.env.PORT || 3001;

// To support multiple simultaneous connections we have a lookup object from
// sessionId to transport
const transports: { [sessionId: string]: SSEServerTransport } = {};

// SSE endpoint for clients to connect
app.get("/sse", async (req: Request, res: Response) => {
  console.log("Client connecting via SSE...");
  // Use '/messages' as the path where clients will POST back
  const transport = new SSEServerTransport("/messages", res);
  const sessionId = transport.sessionId;
  transports[sessionId] = transport;
  console.log(`Transport created with sessionId: ${sessionId}`);

  res.on("close", () => {
    console.log(`Client disconnected: ${sessionId}`);
    delete transports[sessionId];
    // Optional: Clean up any resources associated with the session
  });

  try {
    await server.connect(transport);
    console.log(`MCP Server connected to transport: ${sessionId}`);
  } catch (error) {
    console.error(
      `Error connecting MCP Server to transport ${sessionId}:`,
      error
    );
    // Ensure connection is closed if server.connect fails
    if (!res.closed) {
      res.end();
    }
    delete transports[sessionId];
  }
});

// Endpoint for clients to send messages back to the server
app.post("/messages", async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string;
  if (!sessionId) {
    console.error("Received POST /messages without sessionId query parameter.");
    throw res.status(400).send("Missing sessionId query parameter");
  }

  const transport = transports[sessionId];
  if (transport) {
    console.log(`Received message for sessionId: ${sessionId}`);
    try {
      await transport.handlePostMessage(req, res);
      console.log(`Successfully processed message for sessionId: ${sessionId}`);
    } catch (error) {
      console.error(
        `Error handling POST message for sessionId ${sessionId}:`,
        error
      );
      // handlePostMessage usually sends the response, but if it throws before sending:
      if (!res.headersSent) {
        res.status(500).send("Error processing message");
      }
    }
  } else {
    console.warn(`No active transport found for sessionId: ${sessionId}`);
    res.status(404).send("No active SSE connection found for this sessionId");
  }
});

// Basic root endpoint
app.get("/", (req: Request, res: Response) => {
  res.send("MCP Server with Jina Tools is running. Connect to /sse");
});

app.listen(port, () => {
  console.log(
    `MCP SSE Server with Jina Tools listening on http://localhost:${port}`
  );
  console.log(`SSE endpoint: http://localhost:${port}/sse`);
  console.log(
    `Message endpoint: http://localhost:${port}/messages?sessionId=<sessionId>`
  );
  console.log("Ensure JINA_API_KEY environment variable is set.");
});
