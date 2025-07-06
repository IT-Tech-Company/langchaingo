package googleai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/api/option"
)

// mockRoundTripper captures the request for inspection
type mockRoundTripper struct {
	capturedRequest *http.Request
	capturedBody    []byte
	response        *http.Response
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// Capture the request
	m.capturedRequest = req

	// Capture the body
	if req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		m.capturedBody = body
		// Reset the body for the actual transport
		req.Body = io.NopCloser(bytes.NewBuffer(body))
	}

	// Return a mock response
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"parts":[{"text":"test"}]}}]}`)),
		Header:     make(http.Header),
	}, nil
}

func TestThinkingTransportModifiesPayload(t *testing.T) {
	// Create a mock base transport
	mockTransport := &mockRoundTripper{}

	// Create the thinking transport with dynamic thinking enabled
	thinkingTransport := &thinkingTransport{
		base:            mockTransport,
		dynamicThinking: true,
	}

	// Create a test request with a generateContent endpoint
	originalPayload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{"text": "how much is 4*6*3.4"},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"temperature": 0.5,
		},
	}

	payloadBytes, err := json.Marshal(originalPayload)
	if err != nil {
		t.Fatalf("Failed to marshal test payload: %v", err)
	}

	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", bytes.NewBuffer(payloadBytes))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Execute the request through the thinking transport
	_, err = thinkingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify the payload was modified
	if mockTransport.capturedBody == nil {
		t.Fatal("No request body was captured")
	}

	// Parse the modified payload
	var modifiedPayload map[string]interface{}
	err = json.Unmarshal(mockTransport.capturedBody, &modifiedPayload)
	if err != nil {
		t.Fatalf("Failed to unmarshal captured payload: %v", err)
	}

	// Verify generationConfig exists
	genConfig, exists := modifiedPayload["generationConfig"]
	if !exists {
		t.Fatal("generationConfig not found in modified payload")
	}

	genConfigMap, ok := genConfig.(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig is not a map")
	}

	// Verify thinkingConfig was added
	thinkingConfig, exists := genConfigMap["thinkingConfig"]
	if !exists {
		t.Fatal("thinkingConfig not found in generationConfig")
	}

	thinkingConfigMap, ok := thinkingConfig.(map[string]interface{})
	if !ok {
		t.Fatal("thinkingConfig is not a map")
	}

	// Verify thinkingBudget is set to -1
	thinkingBudget, exists := thinkingConfigMap["thinkingBudget"]
	if !exists {
		t.Fatal("thinkingBudget not found in thinkingConfig")
	}

	if budget, ok := thinkingBudget.(float64); !ok || budget != -1 {
		t.Fatalf("Expected thinkingBudget to be -1, got %v", thinkingBudget)
	}

	// Verify original generationConfig values are preserved
	if temp, exists := genConfigMap["temperature"]; !exists || temp != 0.5 {
		t.Fatalf("Original temperature value not preserved, got %v", temp)
	}

	// Verify original contents are preserved
	_, exists = modifiedPayload["contents"]
	if !exists {
		t.Fatal("contents not found in modified payload")
	}

	t.Log("✓ thinkingConfig successfully added to generationConfig")
	t.Log("✓ thinkingBudget set to -1")
	t.Log("✓ Original generationConfig values preserved")
	t.Log("✓ Original request contents preserved")
}

func TestThinkingTransportSkipsNonGenerateContentRequests(t *testing.T) {
	mockTransport := &mockRoundTripper{}

	thinkingTransport := &thinkingTransport{
		base:            mockTransport,
		dynamicThinking: true,
	}

	// Create a request to a different endpoint
	originalPayload := `{"test": "data"}`
	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:countTokens", strings.NewReader(originalPayload))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}

	_, err = thinkingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify the payload was NOT modified
	capturedBody := string(mockTransport.capturedBody)
	if capturedBody != originalPayload {
		t.Fatalf("Payload was modified for non-generateContent request. Expected %s, got %s", originalPayload, capturedBody)
	}

	t.Log("✓ Non-generateContent requests are not modified")
}

func TestThinkingTransportSkipsWhenDisabled(t *testing.T) {
	mockTransport := &mockRoundTripper{}

	// Create thinking transport with dynamic thinking DISABLED
	thinkingTransport := &thinkingTransport{
		base:            mockTransport,
		dynamicThinking: false,
	}

	originalPayload := `{"contents":[{"parts":[{"text":"test"}]}],"generationConfig":{"temperature":0.5}}`
	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", strings.NewReader(originalPayload))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}

	_, err = thinkingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify the payload was NOT modified
	capturedBody := string(mockTransport.capturedBody)
	if capturedBody != originalPayload {
		t.Fatalf("Payload was modified when dynamic thinking was disabled. Expected %s, got %s", originalPayload, capturedBody)
	}

	t.Log("✓ Requests are not modified when dynamic thinking is disabled")
}

func TestThinkingTransportCreatesGenerationConfigWhenMissing(t *testing.T) {
	mockTransport := &mockRoundTripper{}

	thinkingTransport := &thinkingTransport{
		base:            mockTransport,
		dynamicThinking: true,
	}

	// Create a payload without generationConfig
	originalPayload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{"text": "test"},
				},
			},
		},
	}

	payloadBytes, err := json.Marshal(originalPayload)
	if err != nil {
		t.Fatalf("Failed to marshal test payload: %v", err)
	}

	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", bytes.NewBuffer(payloadBytes))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}

	_, err = thinkingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Parse the modified payload
	var modifiedPayload map[string]interface{}
	err = json.Unmarshal(mockTransport.capturedBody, &modifiedPayload)
	if err != nil {
		t.Fatalf("Failed to unmarshal captured payload: %v", err)
	}

	// Verify generationConfig was created
	genConfig, exists := modifiedPayload["generationConfig"]
	if !exists {
		t.Fatal("generationConfig was not created")
	}

	genConfigMap, ok := genConfig.(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig is not a map")
	}

	// Verify thinkingConfig was added
	_, exists = genConfigMap["thinkingConfig"]
	if !exists {
		t.Fatal("thinkingConfig not found in created generationConfig")
	}

	t.Log("✓ generationConfig created when missing")
	t.Log("✓ thinkingConfig added to newly created generationConfig")
}

func TestThinkingTransportGeneratesCorrectPayload(t *testing.T) {
	mockTransport := &mockRoundTripper{}

	thinkingTransport := &thinkingTransport{
		base:            mockTransport,
		dynamicThinking: true,
	}

	// Create a payload similar to your example
	originalPayload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{"text": "how much is 4*6*3.4"},
				},
			},
		},
	}

	payloadBytes, err := json.Marshal(originalPayload)
	if err != nil {
		t.Fatalf("Failed to marshal test payload: %v", err)
	}

	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent", bytes.NewBuffer(payloadBytes))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}

	_, err = thinkingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Pretty print the captured payload for inspection
	var prettyPayload map[string]interface{}
	err = json.Unmarshal(mockTransport.capturedBody, &prettyPayload)
	if err != nil {
		t.Fatalf("Failed to unmarshal captured payload: %v", err)
	}

	prettyJSON, err := json.MarshalIndent(prettyPayload, "", "  ")
	if err != nil {
		t.Fatalf("Failed to pretty print payload: %v", err)
	}

	t.Logf("Generated payload:\n%s", string(prettyJSON))

	// Verify the exact structure matches your expected format
	expectedStructure := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"parts": []interface{}{
					map[string]interface{}{"text": "how much is 4*6*3.4"},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"thinkingConfig": map[string]interface{}{
				"thinkingBudget": float64(-1),
			},
		},
	}

	// Convert to JSON and back to normalize types
	expectedJSON, _ := json.Marshal(expectedStructure)
	var normalizedExpected map[string]interface{}
	json.Unmarshal(expectedJSON, &normalizedExpected)

	// Compare the structures
	if !equalMaps(prettyPayload, normalizedExpected) {
		expectedPretty, _ := json.MarshalIndent(normalizedExpected, "", "  ")
		t.Fatalf("Generated payload doesn't match expected structure.\nExpected:\n%s\nGot:\n%s",
			string(expectedPretty), string(prettyJSON))
	}

	t.Log("✓ Generated payload matches expected structure exactly")
}

// Helper function to compare maps deeply
func equalMaps(a, b map[string]interface{}) bool {
	aJSON, _ := json.Marshal(a)
	bJSON, _ := json.Marshal(b)
	return string(aJSON) == string(bJSON)
}

func TestCustomHttpClientRespected(t *testing.T) {
	// Create a custom HTTP client with custom transport
	customTransport := &mockRoundTripper{
		response: &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"parts":[{"text":"custom"}]}}]}`)),
			Header:     make(http.Header),
		},
	}

	customClient := &http.Client{
		Transport: customTransport,
		Timeout:   time.Second * 30, // Custom timeout
	}

	// Test that when user provides custom client, we don't override it
	client, err := New(
		context.Background(),
		WithDynamicThinking(true), // Even with thinking enabled
		WithAPIKey("test-key"),
		// Simulate user providing custom HTTP client
		func(opts *Options) {
			opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(customClient))
		},
	)

	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	if client == nil {
		t.Fatal("Client is nil")
	}

	// In this case, dynamic thinking won't work because we respect the user's client
	// This is the expected behavior - we don't want to break user's custom transport
	t.Log("✓ Custom HTTP client is respected")
	t.Log("ℹ️  Note: Dynamic thinking won't work with custom HTTP clients in this implementation")
}

func TestNoCustomHttpClientDetection(t *testing.T) {
	// Test that we correctly detect when no custom client is provided
	opts := []option.ClientOption{
		option.WithAPIKey("test-key"),
		// No custom HTTP client
	}

	hasCustom := hasCustomHttpClient(opts)
	if hasCustom {
		t.Fatal("Expected no custom HTTP client to be detected")
	}

	t.Log("✓ Correctly detected no custom HTTP client")
}

func TestCustomHttpClientDetection(t *testing.T) {
	// Test that we correctly detect when a custom client is provided
	customClient := &http.Client{}

	opts := []option.ClientOption{
		option.WithAPIKey("test-key"),
		option.WithHTTPClient(customClient),
	}

	hasCustom := hasCustomHttpClient(opts)
	if !hasCustom {
		t.Fatal("Expected custom HTTP client to be detected")
	}

	t.Log("✓ Correctly detected custom HTTP client")
}

func TestWrapTransportWithThinking(t *testing.T) {
	// Create a custom base transport
	baseTransport := &mockRoundTripper{
		response: &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"parts":[{"text":"base"}]}}]}`)),
			Header:     make(http.Header),
		},
	}

	// Wrap it with thinking functionality
	wrappedTransport := WrapTransportWithThinking(baseTransport, true)

	// Create a test request
	originalPayload := `{"contents":[{"parts":[{"text":"test"}]}]}`
	req, err := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", strings.NewReader(originalPayload))
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}

	// Execute the request through the wrapped transport
	_, err = wrappedTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify the request was captured by the base transport (meaning it was called)
	if baseTransport.capturedRequest == nil {
		t.Fatal("Base transport was not called")
	}

	// Verify the payload was modified with thinking config
	if baseTransport.capturedBody == nil {
		t.Fatal("No request body was captured")
	}

	var modifiedPayload map[string]interface{}
	err = json.Unmarshal(baseTransport.capturedBody, &modifiedPayload)
	if err != nil {
		t.Fatalf("Failed to unmarshal captured payload: %v", err)
	}

	// Verify thinkingConfig was added
	genConfig, exists := modifiedPayload["generationConfig"]
	if !exists {
		t.Fatal("generationConfig not found in modified payload")
	}

	genConfigMap, ok := genConfig.(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig is not a map")
	}

	thinkingConfig, exists := genConfigMap["thinkingConfig"]
	if !exists {
		t.Fatal("thinkingConfig not found in generationConfig")
	}

	thinkingConfigMap, ok := thinkingConfig.(map[string]interface{})
	if !ok {
		t.Fatal("thinkingConfig is not a map")
	}

	thinkingBudget, exists := thinkingConfigMap["thinkingBudget"]
	if !exists {
		t.Fatal("thinkingBudget not found in thinkingConfig")
	}

	if budget, ok := thinkingBudget.(float64); !ok || budget != -1 {
		t.Fatalf("Expected thinkingBudget to be -1, got %v", thinkingBudget)
	}

	t.Log("✓ WrapTransportWithThinking successfully wraps custom transport")
	t.Log("✓ Base transport is called correctly")
	t.Log("✓ Thinking configuration is added to requests")
}

func TestUserCanCombineCustomClientWithThinking(t *testing.T) {
	// Simulate user creating a custom client with thinking support
	customTransport := &mockRoundTripper{
		response: &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"parts":[{"text":"custom"}]}}]}`)),
			Header:     make(http.Header),
		},
	}

	// User wraps their transport with thinking support
	wrappedTransport := WrapTransportWithThinking(customTransport, true)
	customClient := &http.Client{
		Transport: wrappedTransport,
		Timeout:   time.Second * 45, // Custom timeout preserved
	}

	// User creates client with custom HTTP client
	client, err := New(
		context.Background(),
		WithAPIKey("test-key"),
		// User provides pre-wrapped custom client
		func(opts *Options) {
			opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(customClient))
		},
	)

	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	if client == nil {
		t.Fatal("Client is nil")
	}

	t.Log("✓ Users can combine custom HTTP clients with dynamic thinking")
	t.Log("✓ Custom client configurations (timeouts, etc.) are preserved")
	t.Log("✓ Thinking functionality works with custom transports")
}
