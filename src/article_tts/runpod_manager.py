"""Runpod template and endpoint management via GraphQL API."""

from __future__ import annotations

import os
from typing import Any

import requests
from rich.console import Console

console = Console()


class RunpodManager:
    """Manager for Runpod templates and endpoints via GraphQL API."""

    GRAPHQL_URL = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str | None = None):
        """Initialize the Runpod manager.

        Args:
            api_key: Runpod API key (or set RUNPOD_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Runpod API key required. Set RUNPOD_API_KEY environment "
                "variable or pass it as an argument."
            )

    def _graphql_request(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL request.

        Args:
            query: GraphQL query or mutation
            variables: Optional variables for the query

        Returns:
            Response data

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.GRAPHQL_URL}?api_key={self.api_key}"

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"GraphQL request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        if "errors" in data:
            errors = data["errors"]
            raise RuntimeError(f"GraphQL errors: {errors}")

        return data.get("data", {})

    def get_templates(self) -> list[dict]:
        """Get all templates.

        Returns:
            List of template dictionaries
        """
        query = """
        query {
            myself {
                serverlessTemplates {
                    id
                    name
                    imageName
                    isServerless
                }
            }
        }
        """

        data = self._graphql_request(query)
        return data.get("myself", {}).get("serverlessTemplates", [])

    def get_template_by_name(self, name: str) -> dict | None:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template dictionary or None if not found
        """
        templates = self.get_templates()
        for template in templates:
            if template.get("name") == name:
                return template
        return None

    def create_template(
        self,
        name: str,
        image_name: str,
        docker_args: str = "python handler.py",
        container_disk_gb: int = 10,
        env_vars: dict[str, str] | None = None,
        readme: str = "",
    ) -> str:
        """Create a serverless template.

        Args:
            name: Template name
            image_name: Docker image name (e.g., "your-username/image:tag")
            docker_args: Docker command to run
            container_disk_gb: Container disk size in GB
            env_vars: Environment variables
            readme: Template description

        Returns:
            Template ID
        """
        # Check if template already exists
        existing = self.get_template_by_name(name)
        if existing:
            console.print(
                f"[yellow]Template '{name}' already exists (ID: {existing['id']})[/yellow]"
            )
            return existing["id"]

        env_list = []
        if env_vars:
            env_list = [{"key": k, "value": v} for k, v in env_vars.items()]

        mutation = """
        mutation($input: SaveTemplateInput!) {
            saveTemplate(input: $input) {
                id
                name
                imageName
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "imageName": image_name,
                "dockerArgs": docker_args,
                "containerDiskInGb": container_disk_gb,
                "volumeInGb": 0,  # Serverless templates don't use volumes
                "isServerless": True,
                "env": env_list,
                "readme": readme,
            }
        }

        data = self._graphql_request(mutation, variables)
        template = data.get("saveTemplate", {})
        template_id = template.get("id")

        console.print(f"[green]✓ Created template '{name}' (ID: {template_id})[/green]")
        return template_id

    def get_endpoints(self) -> list[dict]:
        """Get all endpoints.

        Returns:
            List of endpoint dictionaries
        """
        query = """
        query {
            myself {
                endpoints {
                    id
                    name
                    templateId
                    workersMin
                    workersMax
                }
            }
        }
        """

        data = self._graphql_request(query)
        return data.get("myself", {}).get("endpoints", [])

    def get_endpoint_by_name(self, name: str) -> dict | None:
        """Get an endpoint by name.

        Args:
            name: Endpoint name

        Returns:
            Endpoint dictionary or None if not found
        """
        endpoints = self.get_endpoints()
        for endpoint in endpoints:
            if endpoint.get("name") == name:
                return endpoint
        return None

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_ids: str = "AMPERE_16",  # RTX 3060, A2000, A4000
        workers_min: int = 0,
        workers_max: int = 1,
        idle_timeout: int = 5,
        locations: str = "US",
        scaler_type: str = "QUEUE_DELAY",
        scaler_value: int = 4,
    ) -> str:
        """Create a serverless endpoint.

        Args:
            name: Endpoint name
            template_id: Template ID to use
            gpu_ids: GPU type (e.g., "AMPERE_16", "AMPERE_24", "ADA_24")
            workers_min: Minimum workers (0 for serverless)
            workers_max: Maximum workers
            idle_timeout: Idle timeout in seconds
            locations: Deployment locations (e.g., "US", "EU")
            scaler_type: Scaler type ("QUEUE_DELAY", "REQUEST_COUNT")
            scaler_value: Scaler value (delay in seconds or request count)

        Returns:
            Endpoint ID
        """
        # Check if endpoint already exists
        existing = self.get_endpoint_by_name(name)
        if existing:
            console.print(
                f"[yellow]Endpoint '{name}' already exists (ID: {existing['id']})[/yellow]"
            )
            return existing["id"]

        mutation = """
        mutation($input: EndpointInput!) {
            saveEndpoint(input: $input) {
                id
                name
                templateId
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "templateId": template_id,
                "gpuIds": gpu_ids,
                "workersMin": workers_min,
                "workersMax": workers_max,
                "idleTimeout": idle_timeout,
                "locations": locations,
                "scalerType": scaler_type,
                "scalerValue": scaler_value,
                "networkVolumeId": "",
            }
        }

        data = self._graphql_request(mutation, variables)
        endpoint = data.get("saveEndpoint", {})
        endpoint_id = endpoint.get("id")

        console.print(f"[green]✓ Created endpoint '{name}' (ID: {endpoint_id})[/green]")
        return endpoint_id

    def ensure_vibevoice_endpoint(
        self,
        docker_image: str,
        gpu_ids: str = "AMPERE_16",
    ) -> str:
        """Ensure VibeVoice template and endpoint exist.

        Args:
            docker_image: Docker image for VibeVoice
            gpu_ids: GPU type to use

        Returns:
            Endpoint ID
        """
        console.print("[cyan]Checking VibeVoice Runpod setup...[/cyan]")

        # Create or get template
        template_name = "article-tts-vibevoice"
        template_id = self.create_template(
            name=template_name,
            image_name=docker_image,
            docker_args="python /app/handler.py",
            container_disk_gb=10,
            readme="VibeVoice TTS serverless endpoint for article-tts",
        )

        # Create or get endpoint
        endpoint_name = "article-tts-vibevoice-endpoint"
        endpoint_id = self.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_ids=gpu_ids,
            workers_min=0,
            workers_max=1,
            idle_timeout=5,
        )

        console.print(
            f"[green]✓ VibeVoice endpoint ready (ID: {endpoint_id})[/green]"
        )
        return endpoint_id

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete an endpoint.

        Args:
            endpoint_id: Endpoint ID to delete

        Returns:
            True if successful
        """
        mutation = """
        mutation($endpointId: String!) {
            deleteEndpoint(id: $endpointId)
        }
        """

        variables = {"endpointId": endpoint_id}

        try:
            self._graphql_request(mutation, variables)
            console.print(f"[green]✓ Deleted endpoint {endpoint_id}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error deleting endpoint: {e}[/red]")
            return False

    def get_endpoint_status(self, endpoint_id: str) -> dict:
        """Get detailed endpoint status.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Dictionary with endpoint status information
        """
        query = """
        query($endpointId: String!) {
            endpoint(id: $endpointId) {
                id
                name
                workersMin
                workersMax
                pods {
                    id
                    name
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                        gpus {
                            id
                            gpuUtilPercent
                            memoryUtilPercent
                        }
                        container {
                            cpuPercent
                            memoryPercent
                        }
                    }
                    desiredStatus
                    lastStatusChange
                }
            }
        }
        """

        variables = {"endpointId": endpoint_id}

        try:
            data = self._graphql_request(query, variables)
            return data.get("endpoint", {})
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch endpoint status: {e}[/yellow]")
            return {}

    def monitor_endpoint_startup(
        self,
        endpoint_id: str,
        timeout: int = 600,  # 10 minutes
    ) -> bool:
        """Monitor endpoint startup and worker deployment.

        Args:
            endpoint_id: Endpoint ID to monitor
            timeout: Maximum time to wait in seconds

        Returns:
            True if endpoint is ready, False if timeout
        """
        import time
        from rich.live import Live
        from rich.table import Table
        from rich.panel import Panel

        console.print("\n[bold cyan]Monitoring Endpoint Deployment...[/bold cyan]\n")

        start_time = time.time()
        worker_started = False

        try:
            with Live(console=console, refresh_per_second=2) as live:
                while time.time() - start_time < timeout:
                    # Get endpoint status
                    endpoint_data = self.get_endpoint_status(endpoint_id)

                    if not endpoint_data:
                        time.sleep(5)
                        continue

                    # Create status table
                    table = Table(title=f"Endpoint: {endpoint_data.get('name', endpoint_id)}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")

                    elapsed = int(time.time() - start_time)
                    table.add_row("Elapsed Time", f"{elapsed}s")
                    table.add_row("Workers Min/Max", f"{endpoint_data.get('workersMin', 0)}/{endpoint_data.get('workersMax', 0)}")

                    pods = endpoint_data.get("pods", [])
                    table.add_row("Active Pods", str(len(pods)))

                    if pods:
                        for i, pod in enumerate(pods[:3], 1):  # Show up to 3 pods
                            pod_status = pod.get("desiredStatus", "UNKNOWN")
                            runtime = pod.get("runtime", {})
                            uptime = runtime.get("uptimeInSeconds", 0)

                            table.add_row(f"Pod {i} Status", pod_status)
                            table.add_row(f"Pod {i} Uptime", f"{uptime}s")

                            if uptime > 0:
                                worker_started = True

                            # Show GPU/CPU if available
                            gpus = runtime.get("gpus", [])
                            if gpus:
                                gpu = gpus[0]
                                gpu_util = gpu.get("gpuUtilPercent", 0)
                                mem_util = gpu.get("memoryUtilPercent", 0)
                                table.add_row(f"Pod {i} GPU", f"{gpu_util}% util, {mem_util}% mem")

                    # Update status message
                    if worker_started:
                        status_msg = "[green]✓ Worker is running! Container is installing dependencies...[/green]"
                        table.add_row("", "")
                        table.add_row("Status", status_msg)

                        # Wait a bit more for container to finish installing
                        if elapsed > 180:  # After 3 minutes
                            panel = Panel(
                                "[green]Worker deployed successfully![/green]\n\n"
                                "[yellow]Note: First request may take 3-5 minutes as container installs dependencies.[/yellow]",
                                title="Deployment Complete",
                                border_style="green"
                            )
                            live.update(panel)
                            time.sleep(2)
                            return True
                    else:
                        status_msg = "[yellow]Waiting for worker to start...[/yellow]"
                        table.add_row("Status", status_msg)

                    live.update(table)
                    time.sleep(5)

                # Timeout
                console.print("\n[yellow]Monitoring timed out. Endpoint may still be starting.[/yellow]")
                console.print("[cyan]You can check status in Runpod console or try making a request.[/cyan]")
                return False

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring interrupted. Endpoint deployment continues in background.[/yellow]")
            return False
