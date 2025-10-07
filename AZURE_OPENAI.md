# Azure OpenAI Integration

This project now supports Azure OpenAI models in addition to regular OpenAI models. Here's how to set it up and use it.

## Setup

### 1. Environment Variables

Add the following variables to your `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-03-01-preview
```

### 2. Deployment Mapping

The system automatically maps model names to Azure deployment names. The default mapping is:

- `gpt-4.1` → `gpt-41`
- `gpt-4.1-mini` → `gpt-41-mini`
- `gpt-4o` → `gpt-4o`
- `gpt-4o-mini` → `gpt-4o-mini`
- `gpt-35-turbo` → `gpt-35-turbo`
- `gpt-4` → `gpt-4`
- `o1` → `o1`
- `o1-mini` → `o1-mini`
- `o1-pro` → `o1-pro`

You can customize these mappings by editing the `deployment_map` in `src/open_deep_research/utils.py`.

## Usage

### Model Configuration

To use Azure OpenAI models, prefix your model names with `azure_openai:` in your configuration:

```python
# Regular OpenAI
"summarization_model": "openai:gpt-4.1-mini"

# Azure OpenAI  
"summarization_model": "azure_openai:gpt-4.1-mini"
```

### Configuration Examples

```python
from open_deep_research.configuration import Configuration

config = Configuration(
    # Use Azure OpenAI for research
    research_model="azure_openai:gpt-4.1",
    
    # Use Azure OpenAI for summarization
    summarization_model="azure_openai:gpt-4.1-mini",
    
    # Use Azure OpenAI for compression
    compression_model="azure_openai:gpt-4.1",
    
    # Use Azure OpenAI for final report
    final_report_model="azure_openai:gpt-4.1"
)
```

### Environment-based Configuration

You can also set model configurations via environment variables:

```bash
# Use Azure OpenAI models
RESEARCH_MODEL=azure_openai:gpt-4.1
SUMMARIZATION_MODEL=azure_openai:gpt-4.1-mini
COMPRESSION_MODEL=azure_openai:gpt-4.1
FINAL_REPORT_MODEL=azure_openai:gpt-4.1
```

## Features

- **Automatic Environment Setup**: The system automatically configures required Azure OpenAI environment variables
- **Deployment Mapping**: Model names are automatically mapped to Azure deployment names
- **Seamless Integration**: Works with all existing functionality including search tools, MCP tools, and research workflows
- **Token Limit Support**: Azure OpenAI models are included in token limit tracking
- **Error Handling**: Proper error handling for Azure-specific issues

## Supported Models

The following Azure OpenAI models are supported with token limits:

- `azure_openai:gpt-4.1-mini` (1,047,576 tokens)
- `azure_openai:gpt-4.1` (1,047,576 tokens)
- `azure_openai:gpt-4o-mini` (128,000 tokens)
- `azure_openai:gpt-4o` (128,000 tokens)
- `azure_openai:gpt-35-turbo` (16,385 tokens)
- `azure_openai:gpt-4` (128,000 tokens)
- `azure_openai:o1` (200,000 tokens)
- `azure_openai:o1-mini` (200,000 tokens)
- `azure_openai:o1-pro` (200,000 tokens)

## Example Usage

Here's a complete example of using Azure OpenAI with the deep research system:

```python
import asyncio
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration

async def main():
    # Configure to use Azure OpenAI
    config = {
        "configurable": {
            "research_model": "azure_openai:gpt-4.1",
            "summarization_model": "azure_openai:gpt-4.1-mini",
            "compression_model": "azure_openai:gpt-4.1",
            "final_report_model": "azure_openai:gpt-4.1",
            "search_api": "tavily"
        }
    }
    
    # Run research with Azure OpenAI
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": "Research the latest developments in AI"}]},
        config=config
    )
    
    print(result["messages"][-1]["content"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure your Azure OpenAI API key and endpoint are correct
2. **Deployment Not Found**: Verify that your Azure deployment names match the mapping in the code
3. **API Version Issues**: Make sure you're using a supported API version (2025-03-01-preview recommended)

### Custom Deployment Names

If your Azure deployments use different names, update the `deployment_map` in `src/open_deep_research/utils.py`:

```python
deployment_map = {
    "gpt-4.1": "your-custom-gpt-41-deployment",
    "gpt-4.1-mini": "your-custom-gpt-41-mini-deployment",
    # ... other mappings
}
```

### Testing

Run the test script to verify your Azure OpenAI setup:

```bash
python test_azure_openai.py
```

This will test:
- Environment configuration
- API key retrieval
- Model initialization
- Deployment mapping