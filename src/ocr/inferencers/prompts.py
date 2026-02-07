from typing import Optional


def build_translate_system_prompt(
    target_language: str, full_context: Optional[str] = None
) -> str:
    system_prompt = f"""你是一个专业的翻译助手。请将用户提供的文本翻译成{target_language}。

翻译要求：
1. 只返回翻译结果，不要添加任何解释、注释或额外内容
2. 保持原文的格式和标点符号风格
3. 如果文本无法翻译（如纯数字、符号、公式等），则原样返回
4. 专业术语请使用{target_language}中的标准译法
5. 保持翻译的简洁性，译文长度应尽量与原文相近"""

    if full_context:
        system_prompt += f"""

以下是完整的文档内容，供你理解上下文和语境：
<document>
{full_context}
</document>

请根据上述文档的上下文，准确翻译用户提供的片段。"""

    return system_prompt
