import streamlit as st

st.title('Test Streamlit App')
st.write('This is a simple test to check if Streamlit is working correctly.')

user_input = st.text_input('Type something here')
if user_input:
    st.write(f'You typed: {user_input}')