# 06_multi_agent.py —— 多 Agent 流水线命令行入口
#
# 【用法】
#   python 06_multi_agent.py                 跑内置示例问题
#   python 06_multi_agent.py "你的法律问题"    跑指定问题
#
# 【真正的实现在哪】在 multi_agent_runner.py 里。
#   为什么拆开？因为本文件以数字开头，不是合法模块名，别的脚本没法 import 它。
#   所以：能 import 的核心逻辑放 multi_agent_runner.py，命令行入口放这里。
import sys

from multi_agent_runner import run_pipeline

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "我借给朋友十万块，约定了利息，他两年不还，我还能起诉吗？"
    print("=" * 60)
    print(f"问题：{q}")
    print("=" * 60)

    result = run_pipeline(q)

    print("\n" + "=" * 60)
    print("最终答案：")
    print("=" * 60)
    print(result["answer"])
    print(
        f"\n依据法条数：{len(result['sources'])}"
        f"  校验得分：{result['score']}"
        f"  重写次数：{result['retry']}"
    )
