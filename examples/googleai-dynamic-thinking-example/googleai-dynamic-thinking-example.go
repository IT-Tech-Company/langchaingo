// Set the GOOGLE_API_KEY env var to your API key taken from ai.google.dev
// This example demonstrates dynamic thinking functionality with Google AI models
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/IT-Tech-Company/langchaingo/llms"
	"github.com/IT-Tech-Company/langchaingo/llms/googleai"
	"google.golang.org/api/option"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY environment variable is required")
	}

	fmt.Println("=== Google AI Dynamic Thinking Examples ===\n")

	// Example 1: Client-level dynamic thinking
	fmt.Println("1. Client-level dynamic thinking:")
	clientLevelExample(ctx, apiKey)

	// Example 2: Call-level dynamic thinking
	fmt.Println("\n2. Call-level dynamic thinking:")
	callLevelExample(ctx, apiKey)

	// Example 3: Custom HTTP client with dynamic thinking
	fmt.Println("\n3. Custom HTTP client with dynamic thinking:")
	customClientExample(ctx, apiKey)
}

func clientLevelExample(ctx context.Context, apiKey string) {
	// Create client with dynamic thinking enabled for all requests
	llm, err := googleai.New(
		ctx,
		googleai.WithDynamicThinking(true),
		googleai.WithAPIKey(apiKey),
	)
	if err != nil {
		log.Printf("Error creating client: %v", err)
		return
	}

	// Use a thinking model for complex reasoning
	prompt := "Solve this step by step: If a train travels 120 km in 1.5 hours, and then travels another 80 km in 45 minutes, what is the average speed for the entire journey?"

	answer, err := llms.GenerateFromSinglePrompt(
		ctx,
		llm,
		prompt,
		llms.WithModel("gemini-2.5-flash-lite-preview-06-17"),
	)
	if err != nil {
		log.Printf("Error generating response: %v", err)
		return
	}

	fmt.Printf("Question: %s\n", prompt)
	fmt.Printf("Answer with thinking: %s\n", answer)
}

func callLevelExample(ctx context.Context, apiKey string) {
	// Create regular client without thinking
	llm, err := googleai.New(ctx, googleai.WithAPIKey(apiKey))
	if err != nil {
		log.Printf("Error creating client: %v", err)
		return
	}

	// Enable thinking for this specific call
	prompt := "Explain the logic behind this math problem: What's the next number in the sequence 2, 6, 12, 20, 30, ?"

	answer, err := llms.GenerateFromSinglePrompt(
		ctx,
		llm,
		prompt,
		llms.WithModel("gemini-2.5-flash-lite-preview-06-17"),
		llms.WithDynamicThinking(), // Enable thinking for this call only
	)
	if err != nil {
		log.Printf("Error generating response: %v", err)
		return
	}

	fmt.Printf("Question: %s\n", prompt)
	fmt.Printf("Answer with call-level thinking: %s\n", answer)
}

func customClientExample(ctx context.Context, apiKey string) {
	// Create custom HTTP transport with special settings
	customTransport := &http.Transport{
		MaxIdleConns:        50,
		IdleConnTimeout:     60 * time.Second,
		DisableCompression:  false,
		MaxIdleConnsPerHost: 10,
	}

	// Wrap the custom transport with thinking support
	wrappedTransport := googleai.WrapTransportWithThinking(customTransport, true)

	// Create custom HTTP client with the wrapped transport
	customClient := &http.Client{
		Transport: wrappedTransport,
		Timeout:   45 * time.Second, // Custom timeout
	}

	// Create Google AI client with custom HTTP client
	llm, err := googleai.New(
		ctx,
		googleai.WithAPIKey(apiKey),
		func(opts *googleai.Options) {
			opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(customClient))
		},
	)
	if err != nil {
		log.Printf("Error creating client: %v", err)
		return
	}

	prompt := "Design a simple algorithm to find the shortest path between two points in a grid with obstacles. Explain your reasoning."

	answer, err := llms.GenerateFromSinglePrompt(
		ctx,
		llm,
		prompt,
		llms.WithModel("gemini-2.5-flash-lite-preview-06-17"),
	)
	if err != nil {
		log.Printf("Error generating response: %v", err)
		return
	}

	fmt.Printf("Question: %s\n", prompt)
	fmt.Printf("Answer with custom client + thinking: %s\n", answer)
}
