import streamlit as st


def run():
    
    st.header(":mailbox: Contact Us!")


    contact_form = """
    <form action="https://formsubmit.co/aniketdhage1508@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <input type="digit" name="Mobile Number" placeholder="Mobile Number" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send </button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("Style/style.css")
    # Create the footer
    st.header("Find Us")
    col1, col2=st.columns(2)
    # Display the content inside a box
    col1.markdown("""
        <div style='background-color: black; padding: 5px 20px;margin-left:10px'>
            <h6 style='text-align: left;margin-top:15px;'><b>Aniket Dhage</b></h6>
            <ul>
                <li><a href="https://twitter.com/your_twitter_username/" target="_blank">Twitter</a></li>
                <li><a href="https://www.linkedin.com/in/your_linkedin_profile/" target="_blank">LinkedIn</a></li>
                <li>+918530841508</li>
                <li><a href="mailto:aniketdhage1508@gmail.com">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    col2.markdown("""
        <div style='background-color: black; padding: 5px 20px;margin-left:10px'>
            <h6 style='text-align: left;margin-top:15px;'><b>Sanika Butle</b></h6>
            <ul>
                <li><a href="https://twitter.com/your_twitter_username/" target="_blank">Twitter</a></li>
                <li><a href="https://www.linkedin.com/in/your_linkedin_profile/" target="_blank">LinkedIn</a></li>
                <li>+918530841508</li>
                <li><a href="mailto:sanikabutle@gmail.com">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

        

