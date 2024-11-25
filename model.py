# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
import threading

import numpy as np
import triton_python_backend_utils as pb_utils

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline


class TritonPythonModel:

    def initialize(self, args):
        self.logger = pb_utils.Logger

        # Parse model configs
        self.model_config = json.loads(args['model_config'])

        # Ensure the model is in decoupled mode
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)
        assert using_decoupled, 'LMDeploy model should be configured to decoupled mode'

        # Parse parameters
        parameters = self.model_config['parameters']
        model_name = parameters['model_name']['string_value']
        tp = int(parameters['tp']['string_value'])

        # Start LMDeploy engine
        model_path = os.path.join(args['model_repository'],
                                  args['model_version'], 'weights')
        engine_config = TurbomindEngineConfig(tp=tp)
        self.engine = pipeline(model_path=model_path,
                               model_name=model_name,
                               backend_config=engine_config)

        self.request_id = 0

        # Create event loop to process requests asynchronously
        self.event_loop = asyncio.get_event_loop()
        self.engine_thread = threading.Thread(target=self._engine_loop)
        self.shutdown_event = asyncio.Event()
        self.engine_thread.start()

        self.logger.log_info('LMDeploy backend initialized and running.')

    def _engine_loop(self):
        self.logger.log_info('Engine loop started.')
        self.event_loop.run_until_complete(self._await_shutdown())
        self.event_loop.close()
        self.logger.log_info('Engine loop closed.')

    async def _await_shutdown(self):
        # Await the shutdown signal
        await self.shutdown_event.wait()
        self.logger.log_info('Received shutdown signal.')

        # Cancel unfinished tasks
        for task in asyncio.all_tasks(loop=self.event_loop):
            if task is not asyncio.current_task(loop=self.event_loop):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self.logger.log_info('Unfinished task canceled.')

    def _get_optional_configs(self, request):
        optional_configs = {}
        config_names = [
            'temperature', 'top_p', 'top_k', 'stop_words', 'bad_words',
            'repetition_penalty', 'skip_special_tokens'
        ]
        for config_name in config_names:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, config_name)
            if input_tensor is not None:
                if config_name in ['stop_words', 'bad_words']:
                    optional_configs[config_name] = [
                        obj.decode() if isinstance(obj, bytes) else obj
                        for obj in input_tensor.as_numpy().tolist()
                    ]
                else:
                    optional_configs[config_name] = input_tensor.as_numpy().item()
        return optional_configs

    async def _process_request(self, request_id, request):
        response_sender = request.get_response_sender()

        try:
            # Parse request inputs
            prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, 'prompt')
            prompt = prompt_tensor.as_numpy().item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode()

            max_tokens_tensor = pb_utils.get_input_tensor_by_name(
                request, 'max_tokens')
            max_tokens = max_tokens_tensor.as_numpy().item()

            seed_tensor = pb_utils.get_input_tensor_by_name(
                request, 'seed')
            seed = seed_tensor.as_numpy().item()

            ignore_eos_tensor = pb_utils.get_input_tensor_by_name(
                request, 'ignore_eos')
            ignore_eos = ignore_eos_tensor.as_numpy().item()

            stream_tensor = pb_utils.get_input_tensor_by_name(
                request, 'stream')
            stream = stream_tensor.as_numpy().item()

            # Get optional configurations
            optional_configs = self._get_optional_configs(request)
            random_seed = seed if seed else None
            # Create generation configuration
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens,
                ignore_eos=ignore_eos, min_new_tokens=196,
                logprobs=1,  do_sample=True,random_seed=random_seed,# Ensure logprobs are generated
                **optional_configs
            )
            print(f"Gen Config ${gen_config}")
            # Initialize lists to collect outputs
            choices = []

            # Generate outputs asynchronously
            async for output in self.engine.generate(
                messages=prompt,
                session_id=request_id,
                stream_response=stream,
                gen_config=gen_config,
                do_preprocess=False, sequence_start=True, sequence_end=True
            ):

                # Prepare choice entry
                choice_entry = {
                    "text": output.response,
                    "logprobs": {
                        "token_logprobs": [],
                        "tokens": []
                    },
                    "token_ids": []
                }

                # Extract logprobs as floats aligned with token_ids
                if hasattr(output, 'logprobs') and output.logprobs is not None:
                    if isinstance(output.logprobs, list):
                        # Assuming logprobs is a list of dicts
                        for token_id, logprob_dict in zip(output.token_ids, output.logprobs):
                            logprob = logprob_dict.get(token_id, 0.0)
                            choice_entry["logprobs"]["token_logprobs"].append(logprob)
                            choice_entry["logprobs"]["tokens"].append(f"token_id:{token_id}")
                    elif isinstance(output.logprobs, dict):
                        # If logprobs is a single dict
                        for tid in output.token_ids:
                            logprob = output.logprobs.get(tid, 0.0)
                            choice_entry["logprobs"]["token_logprobs"].append(logprob)
                            choice_entry["logprobs"]["tokens"].append(f"token_id:{tid}")
                    else:
                        # Default logprob if format is unexpected
                        choice_entry["logprobs"]["token_logprobs"].extend([0.0] * len(output.token_ids))
                        choice_entry["logprobs"]["tokens"].extend([f"token_id:{tid}" for tid in output.token_ids])

                # Collect token_ids
                if hasattr(output, 'token_ids') and output.token_ids is not None:
                    choice_entry["token_ids"].extend(output.token_ids)

                choices.append(choice_entry)

                if stream:
                    # For streaming mode, send partial responses as they are generated
                    triton_output_tensors = []

                    # Choices tensor as JSON bytes
                    choices_json = json.dumps({"choices": [choice_entry]}).encode('utf-8')
                    choices_tensor = pb_utils.Tensor(
                        'choices',
                        np.array([choices_json], dtype=object)
                    )
                    triton_output_tensors.append(choices_tensor)

                    # Create and send the inference response
                    resp = pb_utils.InferenceResponse(
                        output_tensors=triton_output_tensors
                    )

                    if output.finish_reason is not None:
                        # Mark the response as complete
                        response_sender.send(
                            resp,
                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                    else:
                        # Send partial response
                        response_sender.send(resp)

            if not stream:
                # For non-streaming mode, aggregate all choices and send once
                final_choices = {"choices": choices}
                final_choices_json = json.dumps(final_choices).encode('utf-8')

                # Create output tensors
                choices_tensor = pb_utils.Tensor(
                    'choices',
                    np.array([final_choices_json], dtype=object)
                )

                # Create and send the final inference response
                resp = pb_utils.InferenceResponse(
                    output_tensors=[choices_tensor]
                )
                response_sender.send(
                    resp,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        except Exception as e:
            # Error handling
            self.logger.log_error(f'Error when processing request: {e}')
            error = pb_utils.TritonError(f'Error when processing request: {e}')

            # Create tensors with error information
            error_choices = json.dumps({"choices": []}).encode('utf-8')

            choices_tensor = pb_utils.Tensor(
                'choices',
                np.array([error_choices], dtype=object)
            )

            # Create and send the inference response with error
            resp = pb_utils.InferenceResponse(
                output_tensors=[choices_tensor],
                error=error
            )
            response_sender.send(
                resp,
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
            raise e

    def execute(self, requests):
        for request in requests:
            # Submit each request to the event loop for asynchronous processing
            asyncio.run_coroutine_threadsafe(
                self._process_request(self.request_id, request),
                self.event_loop
            )
            self.request_id += 1
        return None

    def finalize(self):
        self.logger.log_info('Finalizing LMDeploy backend.')
        # Signal the event loop to shut down
        self.event_loop.call_soon_threadsafe(self.shutdown_event.set)
        if self.engine_thread is not None:
            self.engine_thread.join()
            self.engine_thread = None
        self.logger.log_info('LMDeploy backend finalized.')
