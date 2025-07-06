# Google AI LLM Providers

This directory contains langchaingo providers for Google's AI models.

## Available Providers

* **`googleai`**: Provider for Google AI (https://ai.google.dev/)
* **`vertex`**: Provider for GCP Vertex AI (https://cloud.google.com/vertex-ai/)
* **`palm`**: Provider for legacy PaLM models

Both the `googleai` and `vertex` providers give access to Gemini-family
multi-modal LLMs. The code between these providers is very similar; therefore,
most of the `vertex` package is code-generated from the `googleai` package using
a tool:

    go run ./llms/googleai/internal/cmd/generate-vertex.go < llms/googleai/googleai.go > llms/googleai/vertex/vertex.go

## Dynamic Thinking Support

The Google AI provider supports **dynamic thinking** functionality for enhanced reasoning capabilities with thinking models like Gemini 2.5 Pro and Gemini 2.5 Flash.

### Quick Start

```go
import (
    "context"
    "github.com/IT-Tech-Company/langchaingo/llms"
    "github.com/IT-Tech-Company/langchaingo/llms/googleai"
)

// Enable dynamic thinking at client level
client, err := googleai.New(
    context.Background(),
    googleai.WithDynamicThinking(true),
    googleai.WithAPIKey("your-api-key"),
)

// Or enable per call
response, err := client.GenerateContent(
    ctx,
    messages,
    llms.WithDynamicThinking(),
    llms.WithModel("gemini-2.5-flash-lite-preview-06-17"),
)
```

### Usage Patterns

#### 1. Client-Level Dynamic Thinking
Enable thinking for all requests from this client:

```go
client, err := googleai.New(
    ctx,
    googleai.WithDynamicThinking(true),
    googleai.WithAPIKey("your-api-key"),
)
```

#### 2. Call-Level Dynamic Thinking
Enable thinking for specific requests:

```go
response, err := client.GenerateContent(
    ctx,
    messages,
    llms.WithDynamicThinking(),
    llms.WithModel("gemini-2.5-flash-lite-preview-06-17"),
)
```

#### 3. Custom HTTP Client with Dynamic Thinking
For users who need custom HTTP client configurations (timeouts, proxies, etc.) while maintaining thinking functionality:

```go
import (
    "net/http"
    "time"
    "google.golang.org/api/option"
)

// Your custom transport with special settings
customTransport := &http.Transport{
    MaxIdleConns:        100,
    IdleConnTimeout:     90 * time.Second,
    DisableCompression:  true,
    // ... other custom settings
}

// Wrap with thinking support
wrappedTransport := googleai.WrapTransportWithThinking(customTransport, true)

// Create custom client
customClient := &http.Client{
    Transport: wrappedTransport,
    Timeout:   30 * time.Second,
}

// Use custom client with thinking support
client, err := googleai.New(
    ctx,
    googleai.WithAPIKey("your-api-key"),
    func(opts *googleai.Options) {
        opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(customClient))
    },
)
```

### How It Works

When dynamic thinking is enabled, the provider automatically injects the following configuration into the request's `generationConfig`:

```json
{
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": -1
    }
  }
}
```

This enables the model to include its reasoning process in the response, providing stronger reasoning capabilities for complex tasks.

### Supported Models

Dynamic thinking is supported on thinking-capable models including:
- `gemini-2.5-pro`
- `gemini-2.5-flash-lite-preview-06-17`
- Other Gemini 2.5+ models with thinking capabilities

### Important Notes

- **Custom HTTP Clients**: If you provide a custom HTTP client without using `WrapTransportWithThinking()`, dynamic thinking functionality will be disabled to preserve your client's behavior.
- **Performance**: Thinking models may take longer to respond as they generate reasoning steps.
- **API Compatibility**: This feature uses the Google AI API's beta endpoints and thinking configuration.

----

Testing:

The test code between `googleai` and `vertex` is also shared, and lives in
the `shared_test` directory. The same tests are run for both providers.
