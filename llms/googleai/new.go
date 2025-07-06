// package googleai implements a langchaingo provider for Google AI LLMs.
// See https://ai.google.dev/ for more details.
package googleai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"

	"github.com/IT-Tech-Company/langchaingo/callbacks"
	"github.com/IT-Tech-Company/langchaingo/llms"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// GoogleAI is a type that represents a Google AI API client.
type GoogleAI struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             Options
}

// thinkingTransport is a custom HTTP transport that can inject thinking configuration
type thinkingTransport struct {
	base            http.RoundTripper
	dynamicThinking bool
}

// RoundTrip implements the http.RoundTripper interface
func (t *thinkingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// For now, we'll rely on the client-level setting
	// In the future, we can extend this to support per-call configuration
	shouldAddThinking := t.dynamicThinking

	// Only modify requests to generateContent endpoints when dynamic thinking is enabled
	if shouldAddThinking && strings.Contains(req.URL.Path, "generateContent") {
		// Read the original request body
		if req.Body != nil {
			bodyBytes, err := io.ReadAll(req.Body)
			if err != nil {
				return t.base.RoundTrip(req)
			}
			req.Body.Close()

			// Parse the JSON request
			var requestData map[string]interface{}
			if err := json.Unmarshal(bodyBytes, &requestData); err != nil {
				// If we can't parse the JSON, proceed with original request
				req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				return t.base.RoundTrip(req)
			}

			// Add thinking configuration to generationConfig
			if requestData["generationConfig"] == nil {
				requestData["generationConfig"] = make(map[string]interface{})
			}

			if genConfig, ok := requestData["generationConfig"].(map[string]interface{}); ok {
				genConfig["thinkingConfig"] = map[string]interface{}{
					"thinkingBudget": -1,
				}
			}

			// Marshal the modified request back to JSON
			modifiedBody, err := json.Marshal(requestData)
			if err != nil {
				// If we can't marshal, proceed with original request
				req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				return t.base.RoundTrip(req)
			}

			// Create new request body with modified content
			req.Body = io.NopCloser(bytes.NewBuffer(modifiedBody))
			req.ContentLength = int64(len(modifiedBody))
		}
	}

	return t.base.RoundTrip(req)
}

var _ llms.Model = &GoogleAI{}

// New creates a new GoogleAI client.
func New(ctx context.Context, opts ...Option) (*GoogleAI, error) {
	clientOptions := DefaultOptions()
	for _, opt := range opts {
		opt(&clientOptions)
	}
	clientOptions.EnsureAuthPresent()

	gi := &GoogleAI{
		opts: clientOptions,
	}

	// Handle dynamic thinking by adding our HTTP transport
	if clientOptions.DynamicThinking {
		if hasCustomHttpClient(clientOptions.ClientOptions) {
			// User provided a custom HTTP client, but wants dynamic thinking
			// Since we append our client last, it will override the user's client
			// This is a limitation - for custom clients with thinking, users should
			// wrap their own transport with thinkingTransport manually
		}

		// Create HTTP client with thinking transport (this will be the final client used)
		httpClient := &http.Client{
			Transport: &thinkingTransport{
				base:            http.DefaultTransport,
				dynamicThinking: true,
			},
		}

		// Add our HTTP client to the options (last option wins)
		clientOptions.ClientOptions = append(clientOptions.ClientOptions, option.WithHTTPClient(httpClient))
	}

	client, err := genai.NewClient(ctx, clientOptions.ClientOptions...)
	if err != nil {
		return gi, err
	}

	gi.client = client
	return gi, nil
}

// WrapTransportWithThinking wraps an HTTP transport with dynamic thinking support.
// This is useful when users want to provide their own HTTP client while still
// enabling dynamic thinking functionality.
//
// Example usage:
//
//	customTransport := &http.Transport{...}
//	wrappedTransport := googleai.WrapTransportWithThinking(customTransport, true)
//	customClient := &http.Client{Transport: wrappedTransport}
//	client, err := googleai.New(ctx, googleai.WithAPIKey("key"), func(opts *googleai.Options) {
//	    opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(customClient))
//	})
func WrapTransportWithThinking(base http.RoundTripper, enableThinking bool) http.RoundTripper {
	return &thinkingTransport{
		base:            base,
		dynamicThinking: enableThinking,
	}
}

// hasCustomHttpClient checks if the user provided a custom HTTP client
func hasCustomHttpClient(opts []option.ClientOption) bool {
	for _, opt := range opts {
		v := reflect.ValueOf(opt)
		ts := v.Type().String()

		if ts == "option.withHTTPClient" {
			return true
		}
	}
	return false
}
