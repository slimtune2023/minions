from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

from minions.clients import OpenAIClient, TogetherClient

from minions.usage import Usage
from minions.utils.energy_tracking import PowerMonitor
from minions.utils.energy_tracking import cloud_inference_energy_estimate, better_cloud_inference_energy_estimate

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL,
    WORKER_PRIVACY_SHIELD_PROMPT,
    REFORMAT_QUERY_PROMPT,
)


def _escape_newlines_in_strings(json_str: str) -> str:
    # This regex naively matches any content inside double quotes (including escaped quotes)
    # and replaces any literal newline characters within those quotes.
    # was especially useful for anthropic client
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    # Minimal fix: escape newlines only within quoted JSON strings.
    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise


class Minion:
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
        images=None,
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """

        # Setup and start power monitor
        monitor = PowerMonitor(mode="auto", interval=1.0)
        monitor.start()

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Join context sections
        context = "\n\n".join(context)

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
        }

        # Initialize message histories and usage tracking
        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            }
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                "output": None,
            }
        )

        # print whether privacy is enabled
        print("Privacy is enabled: ", is_privacy)

        remote_usage = Usage()
        local_usage = Usage()

        remote_usage_tokens = []
        local_usage_tokens = []

        worker_messages = []
        supervisor_messages = []

        # if privacy import from minions.utils.pii_extraction
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor

            # Extract PII from context
            pii_extractor = PIIExtractor()
            str_context = "\n\n".join(context)
            pii_extracted = pii_extractor.extract_pii(str_context)

            # Extract PII from query
            query_pii_extracted = pii_extractor.extract_pii(task)
            reformat_query_task = REFORMAT_QUERY_PROMPT.format(
                query=task, pii_extracted=str(query_pii_extracted)
            )

            # Clean PII from query
            reformatted_task, usage, done_reason = self.local_client.chat(
                messages=[{"role": "user", "content": reformat_query_task}]
            )
            local_usage += usage
            local_usage_tokens.append((usage.prompt_tokens, usage.completion_tokens))

            pii_reformatted_task = reformatted_task[0]

            # Log the reformatted task
            output = f"""**PII Reformated Task:**
            {pii_reformatted_task}
            """

            if self.callback:
                self.callback("worker", output)

            # Initialize message histories
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(
                        task=pii_reformatted_task
                    ),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                }
            ]
        else:
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                    "images": images,
                }
            ]

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages, response_format={"type": "json_object"}
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages
            )

        remote_usage += supervisor_usage
        remote_usage_tokens.append((supervisor_usage.prompt_tokens, supervisor_usage.completion_tokens))
        
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the last conversation entry with the ouput
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            try:
                supervisor_json = json.loads(supervisor_response[0])

            except:
                supervisor_json = _extract_json(supervisor_response[0])
        else:
            supervisor_json = _extract_json(supervisor_response[0])

        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {"user": "local", "prompt": supervisor_json["message"], "output": None}
        )

        final_answer = None
        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = self.local_client.chat(
                messages=worker_messages
            )

            local_usage += worker_usage
            local_usage_tokens.append((worker_usage.prompt_tokens, worker_usage.completion_tokens))

            if is_privacy:
                if self.callback:
                    output = f"""**_My output (pre-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)

                worker_privacy_shield_prompt = WORKER_PRIVACY_SHIELD_PROMPT.format(
                    output=worker_response[0],
                    pii_extracted=str(pii_extracted),
                )
                worker_response, worker_usage, done_reason = self.local_client.chat(
                    messages=[{"role": "user", "content": worker_privacy_shield_prompt}]
                )
                local_usage += worker_usage
                local_usage_tokens.append((worker_usage.prompt_tokens, worker_usage.completion_tokens))

                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )
                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    output = f"""**_My output (post-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)
            else:
                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    self.callback("worker", worker_messages[-1])

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(response=worker_response[0])

                # Add supervisor COT prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": cot_prompt, "output": None}
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )

                remote_usage += usage
                remote_usage_tokens.append((usage.prompt_tokens, usage.completion_tokens))

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                    response=step_by_step_response[0]
                )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                )
            else:
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages
                )

            remote_usage += supervisor_usage
            remote_usage_tokens.append((supervisor_usage.prompt_tokens, supervisor_usage.completion_tokens))
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                try:
                    supervisor_json = json.loads(supervisor_response[0])
                except:
                    supervisor_json = _extract_json(supervisor_response[0])
            else:
                supervisor_json = _extract_json(supervisor_response[0])

            if supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json["answer"]
                conversation_log["generated_final_answer"] = final_answer
                break
            else:
                next_question = supervisor_json["message"]
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "local", "prompt": next_question, "output": None}
                )

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Log the final result
        if logging_id:
            # use provided logging_id
            log_filename = f"{logging_id}_minion.json"
        else:
            # fall back to timestamp + task abbrev
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        # Stop tracking power
        monitor.stop()

        # Estimate energy consumption over entire conversation/query
        use_better_estimate = True
        
        # Estimate local minion energy consumption
        final_estimates = monitor.get_final_estimates()
        minion_local_energy = float(final_estimates["Measured Energy"][:-2])
        
        # Estimate remote energy consumption (minion and remote-only)
        minion_remote_energy = None
        remote_only_energy = None

        if use_better_estimate:
            minion_remote_energy = 0
            for (input_tokens, output_tokens) in remote_usage_tokens:
                estimate = better_cloud_inference_energy_estimate(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                minion_remote_energy += estimate["total_energy_joules"]
            
            remote_only_energy = minion_remote_energy
            for (input_tokens, output_tokens) in local_usage_tokens:
                estimate = better_cloud_inference_energy_estimate(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                remote_only_energy += estimate["total_energy_joules"]
        
        else:
            # Local/remote input/output tokens
            local_input_tokens = local_usage.prompt_tokens
            local_output_tokens = local_usage.completion_tokens
            remote_input_tokens = remote_usage.prompt_tokens
            remote_output_tokens = remote_usage.completion_tokens

            total_input_tokens = local_input_tokens + remote_input_tokens
            total_output_tokens = local_output_tokens + remote_output_tokens

            # Estimate remote-only energy consumption (remote processes all input/output tokens)
            _, remote_only_energy, _ = cloud_inference_energy_estimate(
                tokens=total_output_tokens+total_input_tokens
            )

            # Estimate minion energy consumption (including both remote and local energy consumption)
            _, minion_remote_energy, _ = cloud_inference_energy_estimate(
                tokens=remote_output_tokens+remote_input_tokens,
            )

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
            "remote_only_energy": remote_only_energy,
            "minion_local_energy": minion_local_energy,
            "minion_remote_energy": minion_remote_energy
        }
