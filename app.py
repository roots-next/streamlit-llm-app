import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser # 回答を文字列で得るため

# --- OpenAI APIキーの読み込み ---
# .envファイルから環境変数を読み込む（ローカル実行時に必要）
# デプロイ環境ではStreamlit Secretsから読み込まれるので、load_dotenv()は不要になるが、
# ローカル開発のために残しておくのが一般的。
load_dotenv()

# 環境変数からAPIキーを読み込む
# Streamlit Community CloudのSecretsで設定した場合もここで読み込まれる
openai_api_key = os.getenv("OPENAI_API_KEY")

# APIキーが設定されていない場合の基本的なエラーハンドリング
if not openai_api_key:
    st.error("OpenAI APIキーが設定されていません。.envファイルを確認するか、Streamlit CloudのSecretsに設定してください。")
    st.stop() # アプリの実行を停止

# --- LLMからの回答を生成する関数 ---
def get_llm_response(user_input, expert_choice):
    """
    ユーザー入力と選択された専門家に基づいてLLMからの回答を生成します。

    Args:
        user_input (str): ユーザーが入力したテキスト。
        expert_choice (str): ラジオボタンで選択された専門家の種類。

    Returns:
        str: LLMからの回答文字列。Noneの場合はエラー発生。
    """
    # 専門家に応じたシステムメッセージを設定
    # --- ここで専門家の種類と役割を自由に定義してください ---
    if expert_choice == "プログラミングの専門家":
        system_message = "あなたは優秀なプログラミングの専門家です。Pythonに関する質問に対して、初心者にも分かりやすく、具体的なコード例を交えて回答してください。"
    elif expert_choice == "マーケティングの専門家":
        system_message = "あなたは経験豊富なマーケティングの専門家です。最新のデジタルトレンドを踏まえ、中小企業向けの具体的で実践的なアドバイスをしてください。"
    elif expert_choice == "料理研究家":
        system_message = "あなたは創造的な料理研究家です。家庭で手軽に作れる、美味しくて健康的なレシピのアイデアを提供してください。材料リストと手順を明確に示してください。"
    # --- 新しい専門家を追加する場合は、elifを追加してください ---
    # 例:
    # elif expert_choice == "旅行プランナー":
    #     system_message = "あなたは経験豊富な旅行プランナーです。予算や興味に合わせて、魅力的な旅行プランを提案してください。"
    else:
        # デフォルト（予期しない選択肢の場合）
        system_message = "あなたは親切なアシスタントです。あらゆる質問に丁寧に答えてください。"
        st.warning(f"選択された専門家 '{expert_choice}' の定義が見つかりません。デフォルトのアシスタントとして応答します。")

    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}") # ユーザーからの入力を変数として埋め込む
    ])

    # LLMモデルを初期化 (APIキーは環境変数から自動で読み込まれる)
    # model_nameはお好みで変更可能 (例: "gpt-4o", "gpt-4-turbo" など)
    # temperatureは回答のランダム性を調整（0に近いほど決定的、1に近いほどランダム）
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

        # LangChainのチェーンを作成 (prompt | llm | parser)
        # StrOutputParserを使うことで、LLMの応答(AIMessageオブジェクト)からテキスト内容だけを抽出
        chain = prompt | llm | StrOutputParser()

        # チェーンを実行して回答を生成
        response = chain.invoke({"input": user_input})
        return response

    except Exception as e:
        st.error(f"LLMとの通信中にエラーが発生しました: {e}")
        # エラーの詳細を知りたい場合は以下のようにログ出力やデバッグを行う
        # print(f"Error details: {e}")
        return None


# --- Streamlit UI部分 ---
st.set_page_config(page_title="LLM専門家チャット", page_icon="🤖") # タブのタイトルとアイコン設定
st.title("🤖 LLM専門家チャットアプリ")
st.write("---")
st.write("相談したい専門家を選び、質問を入力してください。LLMがその専門家になりきって回答します。")

# --- 専門家の選択肢 ---
# ラジオボタンで専門家を選択 (get_llm_response関数内のif文と一致させる)
expert_options = ["プログラミングの専門家", "マーケティングの専門家", "料理研究家"] # 必要に応じて追加・変更
# --- ここでexpert_optionsに専門家を追加したら、上のget_llm_response関数にもelifを追加してください ---

selected_expert = st.radio(
    "相談したい専門家を選んでください:",
    expert_options,
    index=0, # デフォルトで選択される項目 (0はリストの最初の項目)
    horizontal=True # ラジオボタンを横並びにする
)

st.write("---") # 区切り線

# ユーザーからの質問入力フォーム
user_query = st.text_area("質問を入力してください:", height=150, placeholder="例: Pythonでリストの要素を逆順にする方法は？")

# 質問ボタン
if st.button("質問する"):
    if user_query:
        # 入力がある場合のみ処理を実行
        st.write("---")
        st.write(f"**あなたの質問 (相談相手: {selected_expert})**")
        st.info(user_query) # 質問内容を表示

        # LLMからの回答を取得して表示
        st.write(f"**🤖 {selected_expert} からの回答:**")
        with st.spinner("回答を生成中です..."): # 処理中のスピナー表示
            answer = get_llm_response(user_query, selected_expert)
            if answer:
                st.success(answer) # 回答を表示 (successは緑色のボックス)
            # get_llm_response内でエラーが発生した場合、エラーメッセージが表示される
    else:
        # 入力がない場合は警告を表示
        st.warning("⚠️ 質問内容を入力してください。")

st.write("---")
st.caption("Powered by LangChain, OpenAI & Streamlit")