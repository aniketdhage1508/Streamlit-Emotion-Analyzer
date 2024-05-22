import streamlit as st


def run():
    
    st.header(":mailbox: Contact Us!")
    
    person = st.selectbox(label = "Contact Person",options = ["Aniket Dhage","Sanika Butle","Ritika Gadapa","Ashish Deshmukh"])
    if person=="Aniket Dhage":
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
    elif person=="Sanika Butle":
        contact_form = """
        <form action="https://formsubmit.co/sanikabutle@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <input type="digit" name="Mobile Number" placeholder="Mobile Number" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send </button>
        </form>
        """
    elif person=="Ritika Gadapa":
        contact_form = """
        <form action="https://formsubmit.co/ritika.22210542@viit.ac.in" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <input type="digit" name="Mobile Number" placeholder="Mobile Number" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send </button>
        </form>
        """
    elif person=="Ashish Deshmukh":
        contact_form = """
        <form action="https://formsubmit.co/ashish.22211117@viit.ac.in" method="POST">
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
                <li><a href="https://www.linkedin.com/in/aniket-dhage-aa085525a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">LinkedIn</a></li>
                <li><a href="https://x.com/Aniket_S_Dhage" target="_blank">Twitter</a></li>
                <li><a href="mailto:aniketdhage1508@gmail.com">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    col2.markdown("""
        <div style='background-color: black; padding: 5px 20px;margin-left:10px'>
            <h6 style='text-align: left;margin-top:15px;'><b>Sanika Butle</b></h6>
            <ul>
                <li><a href="https://www.linkedin.com/in/sanika-butle-28585825a" target="_blank">LinkedIn</a></li>
                <li><a href="mailto:sanikabutle@gmail.com">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    col1, col2=st.columns(2)
    # Display the content inside a box
    col1.markdown("""
        <div style='background-color: black; padding: 5px 20px;margin-left:10px'>
            <h6 style='text-align: left;margin-top:15px;'><b>Ritika Gadapa</b></h6>
            <ul>
                <li><a href="https://www.linkedin.com/in/ritika-gadapa-96994325b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">LinkedIn</a></li>
                <li><a href="https://x.com/Ritikaaa18" target="_blank">Twitter</a></li>
                <li><a href="mailto:ritika.22210542@viit.ac.in">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    col2.markdown("""
        <div style='background-color: black; padding: 5px 20px;margin-left:10px'>
            <h6 style='text-align: left;margin-top:15px;'><b>Ashish Deshmukh</b></h6>
            <ul>
                <li><a href="https://www.linkedin.com/in/ashish-deshmukh-aa432025b/" target="_blank">LinkedIn</a></li>
                <li><a href="https://x.com/__Ashish22__" target="_blank">Twitter</a></li>
                <li><a href="mailto:ashish.22211117@viit.ac.in">Gmail</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

        

