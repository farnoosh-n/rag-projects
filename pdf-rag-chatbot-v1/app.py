import streamlit as st
import requests

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("🤖 PDF Q&A Chatbot")
st.markdown("---")

user_question = st.text_input("Ask a question about the PDF:")

if st.button("Ask"):
    if user_question:
        with st.spinner("AI is analyzing the document... please wait"):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": user_question}
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get("answer")
                    sources = data.get("sources", [])
                    chunks = data.get("chunks", [])

                    st.success("### Answer:")
                    st.markdown(f"**{answer}**")

                    st.write("### Sources (Pages)")
                    for p in sources:
                        st.write(f"Page {p}")

                    st.write("### Retrieved Context")
                    for chunk in chunks:
                        st.markdown(
                            f"<div style='background-color:#ffff99; padding:10px; border-radius:5px;'>{chunk}</div>",
                            unsafe_allow_html=True
                        )

                else:
                    st.error("Error connecting to the server.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")
