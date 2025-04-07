package agents

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/IT-Tech-Company/langchaingo/callbacks"
	"github.com/IT-Tech-Company/langchaingo/chains"
	"github.com/IT-Tech-Company/langchaingo/schema"
	"github.com/IT-Tech-Company/langchaingo/tools"
)

const _intermediateStepsOutputKey = "intermediateSteps"

// Executor is the chain responsible for running agents.
type Executor struct {
	Agent            Agent
	Memory           schema.Memory
	CallbacksHandler callbacks.Handler
	ErrorHandler     *ParserErrorHandler

	MaxIterations           int
	ReturnIntermediateSteps bool
}

var (
	_ chains.Chain           = &Executor{}
	_ callbacks.HandlerHaver = &Executor{}
)

// NewExecutor creates a new agent executor with an agent and the tools the agent can use.
func NewExecutor(agent Agent, opts ...Option) *Executor {
	options := executorDefaultOptions()
	for _, opt := range opts {
		opt(&options)
	}

	return &Executor{
		Agent:                   agent,
		Memory:                  options.memory,
		MaxIterations:           options.maxIterations,
		ReturnIntermediateSteps: options.returnIntermediateSteps,
		CallbacksHandler:        options.callbacksHandler,
		ErrorHandler:            options.errorHandler,
	}
}

func (e *Executor) Call(ctx context.Context, inputValues map[string]any, _ ...chains.ChainCallOption) (map[string]any, error) { //nolint:lll
	inputs, err := inputsToString(inputValues)
	if err != nil {
		return nil, err
	}
	nameToTool := getNameToTool(e.Agent.GetTools())

	steps := make([]schema.AgentStep, 0)
	for i := 0; i < e.MaxIterations; i++ {
		var finish map[string]any
		steps, finish, err = e.doIteration(ctx, steps, nameToTool, inputs)
		if finish != nil || err != nil {
			return finish, err
		}

		if e.MaxIterations > 2 && i == e.MaxIterations-2 {
			steps = append(steps, schema.AgentStep{
				Observation: "\n Important: Do you have enough data to answer? Provide the final answer \n",
			})
		}
	}

	if e.CallbacksHandler != nil {
		e.CallbacksHandler.HandleAgentFinish(ctx, schema.AgentFinish{
			ReturnValues: map[string]any{"output": ErrNotFinished.Error()},
		}, callbacks.WithExecutedSteps(steps))
	}
	return e.getReturn(
		&schema.AgentFinish{ReturnValues: make(map[string]any)},
		steps,
	), ErrNotFinished
}

func (e *Executor) doIteration( // nolint
	ctx context.Context,
	steps []schema.AgentStep,
	nameToTool map[string]tools.Tool,
	inputs map[string]string,
) ([]schema.AgentStep, map[string]any, error) {
	actions, finish, err := e.Agent.Plan(ctx, steps, inputs)
	if errors.Is(err, ErrUnableToParseOutput) && e.ErrorHandler != nil {
		formattedObservation := err.Error()
		if e.ErrorHandler.Formatter != nil {
			formattedObservation = e.ErrorHandler.Formatter(formattedObservation)
		}
		steps = append(steps, schema.AgentStep{
			Observation: formattedObservation,
		})
		return steps, nil, nil
	}
	if err != nil {
		return steps, nil, err
	}

	if len(actions) == 0 && finish == nil {
		return steps, nil, ErrAgentNoReturn
	}

	if finish != nil {
		if e.CallbacksHandler != nil {
			e.CallbacksHandler.HandleAgentFinish(ctx, *finish, callbacks.WithExecutedSteps(steps))
		}
		return steps, e.getReturn(finish, steps), nil
	}

	for _, action := range actions {
		steps, err = e.checkRepeatedAction(steps, action)
		if err != nil {
			return steps, nil, nil // not returning the error because we're giving the chance to the LLM to write the final answer
		}

		steps, err = e.doAction(ctx, steps, nameToTool, action)
		if err != nil {
			return steps, nil, err
		}
	}

	return steps, nil, nil
}

func (e *Executor) checkRepeatedAction(steps []schema.AgentStep, action schema.AgentAction) ([]schema.AgentStep, error) {
	for _, step := range steps {
		if step.Action.Tool == action.Tool && step.Action.ToolInput == action.ToolInput {
			return append(steps, schema.AgentStep{
				Action:      action,
				Observation: "ATTENTION: you are repeating the same action. Now, you have just 2 options: 1. Write the final answer. 2. Write a different action",
			}), fmt.Errorf("repeated action: %s", action.Tool)
		}
	}

	return steps, nil
}
func (e *Executor) doAction(
	ctx context.Context,
	steps []schema.AgentStep,
	nameToTool map[string]tools.Tool,
	action schema.AgentAction,
) ([]schema.AgentStep, error) {
	if e.CallbacksHandler != nil {
		e.CallbacksHandler.HandleAgentAction(ctx, action)
	}

	tool, ok := nameToTool[strings.ToUpper(action.Tool)]
	if !ok {
		if strings.ToLower(action.Tool) == "none" {
			steps = append(steps, schema.AgentStep{
				Action:      action,
				Observation: "ATTENTION: write the final answer. use the format -> Final Answer: ",
			})

			ctx = context.WithValue(ctx, StepsContextKey, steps)
			return steps, nil
		}

		steps = append(steps, schema.AgentStep{
			Action:      action,
			Observation: fmt.Sprintf("%s is not a valid tool, try another one", action.Tool),
		})

		ctx = context.WithValue(ctx, StepsContextKey, steps)
		return steps, nil
	}

	ctx = context.WithValue(ctx, StepsContextKey, steps)

	observation, err := tool.Call(ctx, action.ToolInput)
	if err != nil {
		return nil, err
	}

	steps = append(steps, schema.AgentStep{
		Action:      action,
		Observation: observation,
	})

	ctx = context.WithValue(ctx, StepsContextKey, steps)

	return steps, nil
}

func (e *Executor) getReturn(finish *schema.AgentFinish, steps []schema.AgentStep) map[string]any {
	if e.ReturnIntermediateSteps {
		finish.ReturnValues[_intermediateStepsOutputKey] = steps
	}

	return finish.ReturnValues
}

// GetInputKeys gets the input keys the agent of the executor expects.
// Often "input".
func (e *Executor) GetInputKeys() []string {
	return e.Agent.GetInputKeys()
}

// GetOutputKeys gets the output keys the agent of the executor returns.
func (e *Executor) GetOutputKeys() []string {
	return e.Agent.GetOutputKeys()
}

func (e *Executor) GetMemory() schema.Memory { //nolint:ireturn
	return e.Memory
}

func (e *Executor) GetCallbackHandler() callbacks.Handler { //nolint:ireturn
	return e.CallbacksHandler
}

func inputsToString(inputValues map[string]any) (map[string]string, error) {
	inputs := make(map[string]string, len(inputValues))
	for key, value := range inputValues {
		valueStr, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrExecutorInputNotString, key)
		}

		inputs[key] = valueStr
	}

	return inputs, nil
}

func getNameToTool(t []tools.Tool) map[string]tools.Tool {
	if len(t) == 0 {
		return nil
	}

	nameToTool := make(map[string]tools.Tool, len(t))
	for _, tool := range t {
		nameToTool[strings.ToUpper(tool.Name())] = tool
	}

	return nameToTool
}
