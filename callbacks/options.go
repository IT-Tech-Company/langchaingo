package callbacks

import (
	"github.com/IT-Tech-Company/langchaingo/schema"
)

type Options struct {
	executedSteps []schema.AgentStep
}

// Option is a function type that can be used to modify the creation of the agents
// and executors.
type Option func(*Options)

// WithExecutedSteps is an option for setting the executed steps of the agent, once it is completed.
// See usage at executor, handling the chain finish.
func WithExecutedSteps(steps []schema.AgentStep) Option {
	return func(co *Options) {
		co.executedSteps = steps
	}
}
