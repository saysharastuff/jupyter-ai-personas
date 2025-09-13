import os
import re
import logging
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message, NewMessage
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools
from agno.team.team import Team
from .fetch_ci_failures import fetch_ci_failures
from .template import PRPersonaVariables, PR_PROMPT_TEMPLATE
from .pr_comment_tool import create_inline_pr_comments

logger = logging.getLogger(__name__)

session = boto3.Session()


class PRReviewPersona(BasePersona):
    # Heartbeat intervals
    FIRST_HEARTBEAT_DELAY = 120
    SECOND_HEARTBEAT_DELAY = 180
    THIRD_HEARTBEAT_DELAY = 300
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="PRReviewer",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized assistant for reviewing pull requests and providing detailed feedback.",
            system_prompt="You are a PR reviewer assistant that helps analyze code changes, provide feedback, and ensure code quality.",
        )

    def initialize_team(self, system_prompt):
        model_id = self.config_manager.lm_provider_params["model_id"]
        github_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not github_token:
            raise ValueError(
                "GITHUB_ACCESS_TOKEN environment variable is not set. Please set it with a plain GitHub personal access token (not GitHub Actions syntax)."
            )

        code_quality = Agent(
            name="code_quality",
            role="Code Quality Analyst",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "Review code quality and analyze CI failures:",
                "1. Get repository and PR information:",
                "   - Extract repo URL and PR number from the request",
                "   - Use GithubTools to fetch PR details",
                "2. ALWAYS check CI failures:",
                "   - MUST call fetch_ci_failures with repo_name and pr_number",
                "   - If failures found, analyze error messages and logs",
                "   - If no failures, mention that CI is passing",
                "   - Include CI status in your final report",
                "3. Review code quality:",
                "   - Code style and consistency",
                "   - Code smells and anti-patterns",
                "   - Complexity and readability",
                "   - Performance implications",
                "   - Error handling and edge cases",
                "4. MUST create inline comments for issues found:",
                "   - For each code issue, IMMEDIATELY call create_inline_pr_comments",
                "   - Use exact file paths from PR changes",
                "   - Use line numbers from the diff",
                "   - Do not just mention issues - CREATE the comments",
                "   - Use the exact format: [{\"path\": \"file.py\", \"position\": 10, \"body\": \"issue description\"}]",
            ],
            tools=[
                GithubTools(),
                fetch_ci_failures,
                create_inline_pr_comments,
                ReasoningTools(),
            ],
        )

        documentation_checker = Agent(
            name="documentation_checker",
            role="Documentation Specialist",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Review documentation completeness and quality:",
                "1. Verify docstrings for new/modified functions and classes",
                "2. Check README updates for new features or changes",
                "3. Verify return value documentation",
                "4. Check for documentation consistency",
            ],
            tools=[],
            markdown=True,
        )

        security_checker = Agent(
            name="security_checker",
            role="Security Analyst",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Perform security analysis of code changes:",
                "1. Check for exposed sensitive information (API keys, tokens, credentials)",
                "2. Identify potential SQL injection vulnerabilities",
                "3. Verify proper input sanitization",
                "4. Check for insecure direct object references",
            ],
            tools=[ReasoningTools()],
            markdown=True,
        )

        gitHub = Agent(
            name="github",
            role="GitHub Specialist",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes",
                "Fetch and process pull request data",
                "Analyze code changes and provide structured feedback",
                "Identify issues that need inline comments:",
                "   - Note specific code issues with file path and line number",
                "   - Report findings to the coordinator for comment posting",
                "Note: Requires a valid GitHub personal access token in GITHUB_ACCESS_TOKEN environment variable",
            ],
            tools=[GithubTools()],
            markdown=True,
        )

        pr_review_team = Team(
            name="pr-review-team",
            mode="coordinate",
            members=[code_quality, documentation_checker, security_checker, gitHub],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Coordinate PR review process with specialized team members:",
                "1. Code Quality Analyst:",
                "   - Review code structure and patterns",
                "   - Check CI status and analyze any failures",
                "   - Keep analysis focused and concise",
                "   - Always create inline comments:",
                "2. Documentation Specialist:",
                "   - Review documentation completeness",
                "   - Focus on critical documentation issues",
                "3. Security Analyst:",
                "   - Check for security vulnerabilities",
                "   - Prioritize high-impact issues",
                "4. GitHub Specialist:",
                "   - FIRST ACTION: Call get_pull_request_changes() with actual repo URL and PR number",
                "   - VERIFY: Show actual PR diff data in response",
                "   - NEVER proceed without real GitHub data",
                "5. CRITICAL - Code Quality Analyst handles inline comments:",
                "   - Code Quality Analyst will create inline comments for issues",
                "   - Other team members should report issues to Code Quality Analyst",
                "   - Focus on identifying and reporting actionable issues",
                "6. Synthesize findings:",
                "   - Combine key insights from all members",
                "   - Focus on actionable items",
                "   - Keep responses concise",
                "Chat history: " + system_prompt,
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=False,
            tools=[
                GithubTools(),
                ReasoningTools(),
            ],
        )

        return pr_review_team

    async def process_message(self, message: Message):

        # Send initial acknowledgment message
        pr_match = re.search(r'github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)', message.body)
        if pr_match:
            repo_name = pr_match.group(1)
            pr_number = pr_match.group(2)
            self.ychat.add_message(
                NewMessage(
                    body=f"Got your request. Processing PR #{pr_number} from repo: {repo_name}",
                    sender=self.id,
                )
            )
  
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]

        history_text = ""

        variables = PRPersonaVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=history_text,
        )

        system_prompt = PR_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[
            0
        ].content

        try:
            team = self.initialize_team(system_prompt)

            # Add periodic heartbeat messages during processing
            import asyncio

            # Flag to stop heartbeat when done
            processing = asyncio.Event()
            processing.set()

            async def heartbeat():
                heartbeat_steps = [
                    (self.FIRST_HEARTBEAT_DELAY, "⏳ Still processing large PR..."),
                    (self.SECOND_HEARTBEAT_DELAY, "⏳ Almost done..."),
                    (self.THIRD_HEARTBEAT_DELAY, "⏳ Taking longer than expected, please wait...")
                ]
                
                try:
                    for delay, message in heartbeat_steps:
                        await asyncio.sleep(delay)
                        if not processing.is_set():
                            break
                        self.ychat.add_message(NewMessage(body=message, sender=self.id))
                except asyncio.CancelledError:
                    logger.debug("Heartbeat task cancelled")
                    raise  # Re-raise to properly terminate the task

            heartbeat_task = asyncio.create_task(heartbeat())

            try:
                response = await asyncio.to_thread(
                    team.run,
                    message.body,
                    stream=False,
                    stream_intermediate_steps=False,
                    show_full_reasoning=False,
                )

                # Stop heartbeat
                processing.clear()
                heartbeat_task.cancel()

                self.ychat.add_message(NewMessage(body=response.content, sender=self.id))

            except Exception as run_error:
                processing.clear()
                heartbeat_task.cancel()
                raise run_error

        except ValueError as e:
            error_message = f"Configuration Error: {str(e)}\nThis may be due to missing or invalid environment variables, model configuration, or input parameters."
            self.ychat.add_message(NewMessage(body=error_message, sender=self.id))

        except Exception as e:
            error_message = f"PR Review Error ({type(e).__name__}): {str(e)}\nAn unexpected error occurred while the PR review team was analyzing your request."
            self.ychat.add_message(NewMessage(body=error_message, sender=self.id))
