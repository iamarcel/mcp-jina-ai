import { z } from 'zod';

// Common schemas
export const TextContentSchema = z.object({
  text: z.string(),
});

export const ImageContentSchema = z.object({
  data: z.string(), // base64 encoded image
  mimeType: z.string(),
});

// Reader schemas
export const ReaderRequestSchema = z.object({
  url: z.string(),
  options: z.enum(['Default', 'Markdown', 'HTML', 'Text', 'Screenshot', 'Pageshot']).optional()
});

export const ReaderResponseSchema = z.object({
  code: z.number(),
  status: z.number(),
  data: z.object({
    title: z.string(),
    description: z.string().optional(),
    url: z.string(),
    content: z.string(),
    images: z.record(z.string()).optional(),
    links: z.record(z.string()).optional(),
    usage: z.object({
      tokens: z.number()
    })
  })
});

export const ReadWebPageSchema = z.object({
  url: z.string(),
  format: z.enum(['Default', 'Markdown', 'HTML', 'Text', 'Screenshot', 'Pageshot']).optional(),
  with_links: z.boolean().optional(),
  with_images: z.boolean().optional(),
  with_generated_alt: z.boolean().optional(),
  no_cache: z.boolean().optional()
});

export const SearchWebSchema = z.object({
  query: z.string(),
  count: z.number().optional().default(5),
  retain_images: z.enum(['none', 'all']).optional().default('none'),
  with_generated_alt: z.boolean().optional().default(true),
  return_format: z.enum(['markdown', 'text', 'html']).optional().default('markdown')
});

export const SearchResponseSchema = z.object({
  code: z.number(),
  status: z.number(),
  data: z.array(z.object({
    title: z.string(),
    description: z.string().optional(),
    url: z.string(),
    content: z.string(),
    images: z.record(z.string()).optional(),
    links: z.record(z.string()).optional(),
    usage: z.object({
      tokens: z.number()
    })
  }))
});

export const GroundingSchema = z.object({
  statement: z.string(),
  deepdive: z.boolean().optional().default(false)
});

export const GroundingReferenceSchema = z.object({
  url: z.string(),
  keyQuote: z.string(),
  isSupportive: z.boolean()
});

export const GroundingResponseSchema = z.object({
  code: z.number(),
  status: z.number(),
  data: z.object({
    factuality: z.number(),
    result: z.boolean(),
    reason: z.string(),
    references: z.array(GroundingReferenceSchema)
  })
});

// Embedding schemas
export const EmbeddingResponseSchema = z.union([
  z.object({
    data: z.array(
      z.object({
        embedding: z.array(z.number()),
        index: z.number().optional(),
        object: z.string().optional()
      })
    ),
    model: z.string().optional(),
    object: z.string().optional(),
    usage: z
      .object({
        total_tokens: z.number().optional(),
        prompt_tokens: z.number().optional()
      })
      .optional()
  }),
  z.object({
    embeddings: z.array(z.array(z.number()))
  })
]);

export type EmbeddingResponse = z.infer<typeof EmbeddingResponseSchema>;

export type ReaderRequest = z.infer<typeof ReaderRequestSchema>;
export type ReaderResponse = z.infer<typeof ReaderResponseSchema>;
export type SearchWebRequest = z.infer<typeof SearchWebSchema>;
export type SearchResponse = z.infer<typeof SearchResponseSchema>;
export type GroundingRequest = z.infer<typeof GroundingSchema>;
export type GroundingResponse = z.infer<typeof GroundingResponseSchema>;
