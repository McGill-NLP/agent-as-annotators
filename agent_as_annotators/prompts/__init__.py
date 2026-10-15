from textwrap import dedent


def format_prompt(instructions: str) -> str:
    return dedent(instructions).strip()


TASK_EXPLORATION_PROMPT_TEMPLATE = """
    You have been instructed to spend a few minutes exploring the websites in order to familiarize yourself 
    with their content and functionalities. When you are done, you should reply to the user with a message
    indicating that you are done exploring the websites: "I am done exploring the websites."

    You have been given the following persona:
    {persona}
"""

TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS = """
    You have been instructed to explore the websites in order to familiarize yourself 
    with their content and functionalities. When you are done, you should reply to the user with a message
    indicating that you are done exploring the websites: "I am done exploring the websites." Make sure to
    explore for at least {min_steps} steps before you stop.

    You have been given the following persona:
    {persona}
"""
EXPLORATION_STRATEGY_INSTRUCTIONS = """
    You are exploring this website to build a map of what it can do. You are not solving a task:
    there is no goal to satisfy, nothing to get right, and no credit for finishing early. You are
    done only when you have covered as much of the site as you can within your step budget.

    Judge every step by how much NEW ground it covers, not by how much progress it makes:

    (1) Breadth of pages. Prefer a page you have not seen over one you have. When a section has
        several sub-pages, visit the ones you have skipped rather than re-reading the landing page.
    (2) Feature surfaces. Each site has distinct areas of functionality -- search, filtering and
        sorting, listings and their detail views, creation and editing forms, settings and
        preferences, account pages, dashboards, history. Touch surfaces you have not touched yet.
    (3) Distinct UI affordances. Exercise different kinds of control, not the same kind repeatedly:
        text inputs, dropdowns and selects, checkboxes and toggles, tabs, pagination, sort headers,
        modals, expandable panels, date and range pickers.
    (4) Depth over repetition. Going one level deeper into a section you have only glanced at is
        worth more than another pass over a section you already know. If a page reveals a control
        you do not understand, use it and observe what changes.
    (5) Recoverable states only. Do not delete, purchase, or submit anything you cannot undo. Fill
        a form to see its fields and its validation without committing the result.

    Avoid the two ways this goes wrong: looping between the same few pages, and drilling so far into
    one corner that the rest of the site goes unseen. If you find yourself back where you were,
    treat that as a signal to jump to an untouched surface.

    In your strategy, keep an explicit account of which surfaces you have already covered and name
    the next uncovered one you intend to reach. Recording coverage is what makes the later part of
    the trajectory more informed than the earlier part.

    When you have covered what you can, reply to the user with exactly this message:
    "I am done exploring the websites."
"""

TASK_EXPLORATION_PROMPT_TEMPLATE_COVERAGE = """
    You have been instructed to explore the websites in order to familiarize yourself
    with their content and functionalities. When you are done, you should reply to the user with a message
    indicating that you are done exploring the websites: "I am done exploring the websites." Make sure to
    explore for at least {min_steps} steps before you stop.

    {strategy_instructions}

    You have been given the following persona:
    {persona}
"""

WEBARENA_ANNOTATOR_INSTRUCTIONS = """
    (1) The intent should be abstract and high-level, implying that the task cannot be fulfilled with
    merely one or two actions. As an example, instead of "click the science subreddit", we
    encourage you to come up with something more complex like "post a greeting message
    on science subreddit”, which involves performing multiple actions.
    (2) The intent should be creative. Common tasks such as account creation can be easily thought of.
    We encourage you to add constraints (e.g., "create a Reddit account identical to my
    GitLab one") to make the intents more unique.
    (3) The intent should be formulated as a template by making replaceable elements as variables.
    You are also responsible for developing several instantiations for each variable.
    For example, the intent "create a Reddit account identical to my GitLab one" can be converted
    into "create a {{site1}} account identical to my {{site2}} one", with an instantiation like "{site1:
    Reddit, site2: GitLab}" and another like "{site1: GitLab, site2: OneStopShopping}". Notably,
    tasks derived from the same template can have distinct execution traces. The similarity resides
    primarily in the high-level semantics rather than the specific implementation.
"""

TASK_CREATION_PROMPT_TEMPLATE = """
    You have been recruited as an annotator to design tasks for autonomous agents, which will use the 
    web browser directly to operate on several websites.

    You will be given a persona, instructions, and subsequently the state of a webpage retrieved from 
    your own exploration. The task should be relevant to the persona and webpage state while following
    the instructions.

    You have been given the following persona:
    {persona}

    {annotator_instructions}
    
    {task_examples}

    Return the task in JSON format with the following keys: template, intent, and evaluation_description.
    The evaluation_description should be a description of what the agent should do to properly complete
    the task, which will be used to evaluate the agent's performance.
"""

TASK_EXAMPLES = """
    You have been given the following task examples:
    Information Seeking
    - When was the last time I bought shampoo
    - Compare walking and driving time from AMC Waterfront to Randyland

    Site Navigation
    - Checkout merge requests assigned to me
    - Show me the ergonomic chair with the best rating

    Content & Configuration
    - Post to ask “whether I need a car in NYC”
    - Delete the reviews from the scammer Yoke
"""

TASK_INTENT_PROMPT_TEMPLATE = """
    Based on the conversation above, come up with a {num_intents} tasks following these instructions:
    {annotator_instructions}
    Design tasks that can be accomplished in less than 30 steps.
    Enclose each task intent with tag <intent>...</intent>, followed <eval>...</eval> explaining
    how to evaluate whether the task is completed successfully, what webpages to visit, what information to look for,
    and how it is relevant to the conversation above, followed by an <end> tag, and finally, an <instantiation>...</instantiation> tag
    containing the instantiation of the task intent, in a dict format where the keys are the variable names and the values are the instantiations.
    For multiple intents, you will output a sequence of such tags, e.g.,
    <intent>...</intent><eval>...</eval><instantiation>...</instantiation><end><intent>...</intent><eval>...</eval><instantiation>...</instantiation>.
    Note: for website names, use the general format: __<SITE_NAME>__, i.e., __REDDIT__, __GITLAB__, __SHOPPING__, __MAP__, __WIKIPEDIA__,
    __SHOPPING_ADMIN__. For example, https://self-hosted.example-server.com/my/gitlab/account/ will be __GITLAB__/my/gitlab/account/.
"""

TASK_EXPLORATION_PROMPT_TEMPLATE = format_prompt(TASK_EXPLORATION_PROMPT_TEMPLATE)
WEBARENA_ANNOTATOR_INSTRUCTIONS = format_prompt(WEBARENA_ANNOTATOR_INSTRUCTIONS)
TASK_CREATION_PROMPT_TEMPLATE = format_prompt(TASK_CREATION_PROMPT_TEMPLATE)
TASK_EXAMPLES = format_prompt(TASK_EXAMPLES)
TASK_INTENT_PROMPT_TEMPLATE = format_prompt(TASK_INTENT_PROMPT_TEMPLATE)
TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS = format_prompt(TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS)
EXPLORATION_STRATEGY_INSTRUCTIONS = format_prompt(EXPLORATION_STRATEGY_INSTRUCTIONS)
TASK_EXPLORATION_PROMPT_TEMPLATE_COVERAGE = format_prompt(TASK_EXPLORATION_PROMPT_TEMPLATE_COVERAGE)

# The stop string the `string_match` evaluator terminates the exploration episode
# on. Every exploration template must contain it verbatim; changing it silently
# would leave episodes running to the step cap with no way to stop legitimately.
EXPLORATION_STOP_MESSAGE = "I am done exploring the websites."


def build_exploration_prompt(persona: str, min_steps: int) -> str:
    """Build the Exploration v2 goal: coverage framing + persona + step floor."""
    return TASK_EXPLORATION_PROMPT_TEMPLATE_COVERAGE.format(
        min_steps=min_steps,
        strategy_instructions=EXPLORATION_STRATEGY_INSTRUCTIONS,
        persona=persona,
    )
