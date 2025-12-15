## Themes
```
Generate {{n}} "themes" for a diverse array of decision theory problems. 
- Each theme should be a short sentence. 
- Themes are not fully specified problems, but general types of problems which could be adapted to a variety of more concrete scenarios.
- Themes should be compatible with one or two player single-move games with no opportunity for observations of the opponent move/state. For instance, one-shot prisoners' dilemmas, Newcomblike problems, and smoking-lesion problems are appropriate, whereas iterated bargaining, auctions, and Parfit's hitchiker are not.
- Alternate between whether your themes involve two competitors, or a competitor and some kind of predictive variable. In other words, odd-numbered themes should be vaguely prisoners' dilemma-like, and even-numbered themes should be vaguely Newcomb-like, but you have great latitude in how you interpret this and you should not narrow your thinking. Think outside the box.
- Strive for originality and avoid replicating well-known problems in your response, including any mentioned in this prompt. Hidden biases will ruin your output's value as a dataset.
- Format as a multi-line Python list of strings. You must number every single line with a # comment to keep track of line numbers. Do not use any codeblock formatting. Do not stop until you hit {{n}} themes exactly.
- Your output length limit is 128K tokens, which is more than enough to accommodate this request. You have plenty of room to think and generate.
```
Model: gpt-5.1, text.format: text, effort: high, verbosity: medium

## Schemas
```
# Task
Generate diverse schemas for decision theory problems to outline a new ML testing set.

1. You will be presented with a number of themes: overall loose ideas for decision theory problems.
2. Then, for each theme you will generate three schemas, for the topologies A, B, and C respectively. These schemas are variants of the theme for which different decision theories might give different recommendations.

# Topology of problems
## Shared structure
In all problems, these are the variables and the roles they play.
e: environmental variable. In topology A, it explains why a and s both depend on f (for instance, shared programming). In topology B, it is merely correlated with both f and s (for instance, a genetic mutation affecting the brain and external state). In topology C, it determines how s is determined but is not linked to f (for instance, the sampling of a random stranger).
f: a variable representing what FDT recommends the player do. FDT agents intervene on this variable. It is implicit in every schema so you do not need to provide a description of it.
a: variable for the player action. CDT agents intervene on this variable. EDT agents track all correlations with this variable.
s: another state, which jointly with a determines outcome. Could be an action of another player, a prediction, some feature of the environment, etc.
o: outcome. Presented as a list of payoffs for player 1. In schemas involving another player, payoffs are assumed to be symmetric.

## Topologies
### Topology A:
DAG: e → f; f → a, s; a, s → o
Decision theories: CDT treats action and state as independent; FDT and EDT do not
Desired verdicts: CDT recommends one action, FDT and EDT recommend a different action
Example: Newcomb's problem
{
	"e": "deterministic programming",
	"a": ["one-box", "two-box"],
	"s": ["predict one-box", "predict two-box"],
	"o": [1000, 0, 1100, 100],
	"CDT": 1,
	"FDT": 0,
	"EDT": 0
}

### Topology B:
DAG: e → f, s; f → a; a, s → o
Decision theories: CDT and FDT treat action & state as independent; EDT does not
Desired verdicts: CDT & FDT recommend one action, EDT recommends a different action
Example: smoking-lesion
{
	"e": "smoking gene",
	"a": ["smoke", "abstain"],
	"s": ["cancer", "healthy"],
	"o": [-99, 1, -100, 0],
	"CDT": 0,
	"FDT": 0,
	"EDT": 1,
}

### Topology C:
DAG: e → s; f → a; a, s → o
Decision theories: CDT, FDT, and EDT treat action and state as independent
Desired verdicts: All three decision theories recommend the same action
Example: prisoner's dilemma against a stranger
{
	"e": "random stranger",
	"a": ["cooperate", "cheat"],
	"s": ["cooperate", "cheat"],
	"o": [2, -1, 3, 0],
	"CDT": 1,
	"FDT": 1,
	"EDT": 1
}

# Rules
- Games should be one-shot and involve no further observations on the part of the player beyond the initial setup. For instance, Parfit's hitchhiker, in which the player chooses whether or not to pay after observing whether they get rescued, is not a valid instance of the above topologies because it would draw a direct line from s to a via observation.
- All of the themes and schemas should assume that the player is a deterministic agent of some kind. However, other players may or may not be at your discretion.
- Outcomes should be in order [a[0]s[0], a[0]s[1], a[1]s[0], a[1]s[1]]. That is, the first item in the outcome array should be the payoff for action 0, state 0; the second item should be the payoff for action 0, state 1; and so on.
- Do not copy well-known decision theory problems like Newcomb's problem, twin PD, or smoking-lesion, and do not copy any of the examples in this prompt.
- Themes are suggestions, not strict rules. If a theme is a poor fit to the schema requirements, feel free to modify it as needed to comply with the above rules.

# Themes
Your themes are: {{themes}}
```
Model: gpt-5.1, text.format: json_schema, effort: high, verbosity: medium

JSON schema:
```
{
  "name": "decision_theory_problems",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "problems": {
        "type": "array",
        "description": "List of decision theory problems, each with theme and three topologies (A, B, C).",
        "items": {
          "type": "object",
          "properties": {
            "theme": {
              "type": "string",
              "description": "The theme of the decision theory problem."
            },
            "schema": {
              "type": "object",
              "properties": {
                "A": {
                  "$ref": "#/$defs/topology"
                },
                "B": {
                  "$ref": "#/$defs/topology"
                },
                "C": {
                  "$ref": "#/$defs/topology"
                }
              },
              "required": [
                "A",
                "B",
                "C"
              ],
              "additionalProperties": false
            }
          },
          "required": [
            "theme",
            "schema"
          ],
          "additionalProperties": false
        }
      }
    },
    "required": [
      "problems"
    ],
    "additionalProperties": false,
    "$defs": {
      "topology": {
        "type": "object",
        "properties": {
          "e": {
            "type": "string",
            "description": "The environment variable."
          },
          "a": {
            "type": "array",
            "description": "List of possible actions.",
            "items": {
              "type": "string"
            }
          },
          "s": {
            "type": "array",
            "description": "List of possible states.",
            "items": {
              "type": "string"
            }
          },
          "o": {
            "type": "array",
            "description": "List of 4 integers representing outcomes for each action-state combination.",
            "items": {
              "type": "integer"
            },
            "minItems": 4,
            "maxItems": 4
          },
          "CDT": {
            "type": "integer",
            "description": "Index of the action recommended by CDT (0 or 1).",
            "minimum": 0,
            "maximum": 1
          },
          "FDT": {
            "type": "integer",
            "description": "Index of the action recommended by FDT (0 or 1).",
            "minimum": 0,
            "maximum": 1
          },
          "EDT": {
            "type": "integer",
            "description": "Index of the action recommended by EDT (0 or 1).",
            "minimum": 0,
            "maximum": 1
          }
        },
        "required": [
          "e",
          "a",
          "s",
          "o",
          "CDT",
          "FDT",
          "EDT"
        ],
        "additionalProperties": false
      }
    }
  }
}
```

## Descriptions
```
# Task
Take schemas for decision theory problems and write complete textual descriptions of them.

# How schemas work
## Shared structure
In all problems, these are the variables and the roles they play.
e: environmental variable. In topology A, it explains why a and s both depend on f (for instance, shared programming). In topology B, it is merely correlated with both f and s (for instance, a genetic mutation affecting the brain and external state). In topology C, it determines how s is determined but is not linked to f (for instance, the sampling of a random stranger).
f: a variable representing what FDT recommends the player do. FDT agents intervene on this variable. It is implicit in every topology, so it may not be explicitly mentioned in the schema.
a: variable for the player action. CDT agents intervene on this variable. EDT agents track all correlations with this variable.
s: another state, which jointly with a determines outcome. Could be an action of another player, a prediction, some feature of the environment, etc.
o: outcome. Presented as a list of payoffs for player 1. In schemas involving another player, payoffs are assumed to be symmetric.

## Topologies
### Topology A:
DAG: e → f; f → a, s; a, s → o
Decision theories: CDT treats action and state as independent; FDT and EDT do not
Desired verdicts: CDT recommends one action, FDT and EDT recommend a different action
Example: Newcomb's problem
{
	"e": "deterministic programming",
	"a": ["one-box", "two-box"],
	"s": ["predict one-box", "predict two-box"],
	"o": [1000, 0, 1100, 100],
	"CDT": 1,
	"FDT": 0,
	"EDT": 0
}

### Topology B:
DAG: e → f, s; f → a; a, s → o
Decision theories: CDT and FDT treat action & state as independent; EDT does not
Desired verdicts: CDT & FDT recommend one action, EDT recommends a different action
Example: smoking-lesion
{
	"e": "smoking gene",
	"a": ["smoke", "abstain"],
	"s": ["cancer", "healthy"],
	"o": [-99, 1, -100, 0],
	"CDT": 0,
	"FDT": 0,
	"EDT": 1,
}

### Topology C:
DAG: e → s; f → a; a, s → o
Decision theories: CDT, FDT, and EDT treat action and state as independent
Desired verdicts: All three decision theories recommend the same action
Example: prisoner's dilemma against a stranger
{
	"e": "random stranger",
	"a": ["cooperate", "cheat"],
	"s": ["cooperate", "cheat"],
	"o": [2, -1, 3, 0],
	"CDT": 1,
	"FDT": 1,
	"EDT": 1
}

# Rules
- For each topology instance you should write a complete standalone description. Do not reference previously written problems, nor should you assume that the theme will appear above. Each description should make total sense in isolation, providing the full context necessary without the other problems or theme.
- Each description should be a short, concise paragraph. Use as few sentences as possible, while still conveying all the important points.
- Your description should make the following relationships clear to a careful reader:
	1. Altering a cannot causally alter s.
	2. In A topologies, s subjunctively depends on a, but this is not the case in B and C topologies.
	3. In C topologies, a and s aren't even correlated.
	However, your descriptions should not use the phrase "subjunctive dependence" anywhere, and it should use a variety of language to indicate these relationships.
- Descriptions should be written in the second person.
- Games should be one-shot and involve no further observations on the part of the player beyond the initial setup. For instance, Parfit's hitchhiker, in which the player chooses whether or not to pay after observing whether they get rescued, is not a valid instance of the above topologies because it would draw a direct line from s to a via observation.
- All of the themes and schemas should assume that the player is a deterministic agent of some kind.
- Explicit outcomes numbers must be mentioned in the output.
- Outcomes are in order [a[0]s[0], a[0]s[1], a[1]s[0], a[1]s[1]]. That is, the first item in the outcome array is the payoff for action 0, state 0; the second item should be the payoff for action 0, state 1; and so on.

# Common fail states (avoid)
- Making descriptions reference other descriptions, rather than being standalone
- Making descriptions assume the reader has read the theme, rather than providing all context
- Failing to adhere to schemas
- Failing to make the underlying relationships clear
- Using the same blatant signals for every A, B, and C topology, making it easy for models to overfit to particular word cues rather than picking up underlying problem structure
- Overly long paragraphs
- Using incomplete sentences, vague language, or token contractions to force paragraphs to use fewer words.

# Schemas
Your schemas are: 
{{schemas}}
```

Model: gpt-5.1, text.format: json_schema, effort: low, verbosity: low,

JSON schema:
```
{
  "name": "descriptions_list",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "descriptions": {
        "type": "array",
        "description": "A list of description objects for different versions of a problem sharing a theme.",
        "items": {
          "type": "object",
          "properties": {
            "theme": {
              "type": "string",
              "description": "A verbatim quote of the theme from the input."
            },
            "A": {
              "type": "string",
              "description": "Short paragraph description of problem, version A."
            },
            "B": {
              "type": "string",
              "description": "Short paragraph description of problem, version B."
            },
            "C": {
              "type": "string",
              "description": "Short paragraph description of problem, version C."
            }
          },
          "required": [
            "theme",
            "A",
            "B",
            "C"
          ],
          "additionalProperties": false
        }
      }
    },
    "required": [
      "descriptions"
    ],
    "additionalProperties": false
  }
}
```

