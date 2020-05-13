import itertools
import time
import os
from selenium import webdriver
from bs4 import BeautifulSoup

website_url = "http://146.169.52.15:8000/"
login_user = "slave"
login_pw = "slave"


class WebsiteTester:
    """
    This class contains functions to call the individual functionalities of the website.
    The main user workflows (lookup and upload) are tested in the full_lookup_workflow and full_upload_workflow functions.
    The required files to be uploaded are passed in via file path, please place them in the same folder as this script.
    """

    def __init__(self, website_url, login_user, login_pw, verbose=False, headless=False):
        self.verbose = verbose
        self.login_user = login_user
        self.login_pw = login_pw
        self.driver = self.start_driver(website_url, headless)

    def start_driver(self, website_url, headless):
        # This is the webdriver for Chrome 81, if this is not your Chrome version please
        # download your version from the Selenium website
        if headless:
            op = webdriver.ChromeOptions()
            op.add_argument('headless')
            driver = webdriver.Chrome(os.path.join(os.getcwd(), "chromedriver"), options=op)
        else:
            driver = webdriver.Chrome(os.path.join(os.getcwd(), "chromedriver"))
        driver.get(website_url)
        if self.verbose:
            print("Debug: Driver started.")
        return driver

    def xpath_soup(self, element):
        """
        Generate xpath of soup element
        :param element: bs4 text or node
        :return: xpath as string
        """
        components = []
        child = element if element.name else element.parent
        for parent in child.parents:
            """
            @type parent: bs4.element.Tag
            """
            previous = itertools.islice(parent.children, 0, parent.contents.index(child))
            xpath_tag = child.name
            xpath_index = sum(1 for i in previous if i.name == xpath_tag) + 1
            components.append(xpath_tag if xpath_index == 1 else '%s[%d]' % (xpath_tag, xpath_index))
            child = parent
        components.reverse()
        return '/%s' % '/'.join(components)

    def home(self):
        if self.verbose:
            print("Debug: Accessing Home.")
        self.driver.find_element_by_link_text("Homepage").click()
        return True

    def login(self):
        # Click on Login
        start = time.time()
        bs_start = BeautifulSoup(self.driver.page_source, "lxml")
        bs_navbar = bs_start.find("div", class_="nav-wrapper imperial blue")
        bs_login_button = bs_navbar.find("a", text="Login")

        # if bs_login_button:
            # Click login button in navbar
        if self.verbose:
            print("Debug: Logging in.")
        self.driver.find_element_by_link_text("Login").click()
        self.driver.find_element_by_name("username").send_keys(self.login_user)
        self.driver.find_element_by_name("password").send_keys(self.login_pw)
        bs_login = BeautifulSoup(self.driver.page_source, "lxml")
        bs_login_button = bs_login.find("button", text="Login")
        self.driver.find_element_by_xpath(self.xpath_soup(bs_login_button)).click()
        bs_login = BeautifulSoup(self.driver.page_source, "lxml")
        if "Invalid" in bs_login.text:
            print("Error: LOGIN FAILED - Please check credentials.")
            return False
        # else:
        #     print("Warning: Already logged in.")
        if self.verbose:
            print("Debug: Logged in.")
        return time.time() - start

    def lookup(self, session_id, is_display_MRI, is_from_home):
        start = time.time()
        session_id = str(session_id)
        # Accessing from the home website
        if is_from_home:
            if self.verbose:
                print("Debug: Accessing Lookup from home.")
            bs_home = BeautifulSoup(self.driver.page_source, "lxml")
            assert bs_home.find("p", text="Look-up session IDs"), "Tried to access Lookup from Home but not on Home."
            # Click on Lookup
            self.driver.find_element_by_xpath("/html/body/main/div/div[2]/div[1]/div/div[2]/a").click()
        # Access from ribbon
        else:
            if self.verbose:
                print("Debug: Accessing Lookup from ribbon.")
            self.driver.find_element_by_link_text("Look-up").click()

        # If not logged in
        bs_lookup = BeautifulSoup(self.driver.page_source, "lxml")
        if "You need to be logged in" in bs_lookup.text:
            print("Error: Not logged in. Please login to access Lookup")
            return False

        # Click on the dropdown
        bs_dropdown = bs_lookup.find("input", class_="select-dropdown dropdown-trigger")
        self.driver.find_element_by_xpath(self.xpath_soup(bs_dropdown)).click()
        bs_lookup = BeautifulSoup(self.driver.page_source, "lxml")
        bs_dropdown = bs_lookup.find("form", {"id": "lookup-form"})
        bs_sessions = bs_dropdown.find_all("li")

        found_session = False
        for session in bs_sessions:
            # If the session is available
            if session_id == session.find("span").text:
                # Click on the session ID
                self.driver.find_element_by_xpath(self.xpath_soup(session)).click()
                found_session = True
                break

        if not found_session:
            print("Error: The session ID you specified is not available.")
            return False

        if is_display_MRI:
            if self.verbose:
                print("Debug: MRI slices requested - checking.")
            # Is there a checkbox to be clicked
            bs_checkbox = bs_lookup.find("input", {"id": "display-mri"})
            if bs_checkbox["disabled"] == "disabled":
                print("Warning: Display MRI is not available for the selected session ID")
            else:
                self.driver.find_element_by_xpath(
                    self.xpath_soup(bs_lookup.find("span", text="Display raw MRI slices?"))).click()
                if self.verbose:
                    print("Debug: MRI slices requested - successful.")

        self.driver.find_element_by_id("look-up-btn").click()
        if self.verbose:
            print("Debug: Looking up.")

        # Make sure the MRI slices generated
        if is_display_MRI:
            bs_results = BeautifulSoup(self.driver.page_source, "lxml")
            if not bs_results.find("canvas", {"id": "3Dviewer"}):
                print("Warning: The MRI segment slices are missing even though the option was checked.")
        if self.verbose:
            print("Debug: Lookup successful.")
        return time.time() - start

    def predict(self, wait_max=100):
        self.driver.find_element_by_id("predict").click()
        start = time.time()
        time_expired = 0
        while time_expired < wait_max:
            bs_results = BeautifulSoup(self.driver.page_source, "lxml")
            if bs_results.find("table", {"id": "result_table"}):
                if self.verbose:
                    print("Debug: Prediction results found.")
                return time.time() - start
            time_expired = time.time() - start
        print(f"ERROR: Results table was not generated from prediction after waiting {wait_max} seconds.")
        return time.time() - start


    def segment(self, wait_max=100):
        self.driver.find_element_by_id("segment").click()
        start = time.time()
        time_expired = 0
        while time_expired < wait_max:
            bs_results = BeautifulSoup(self.driver.page_source, "lxml")
            if bs_results.find("option", {"value": "PointData:Predicted Labels"}):
                if self.verbose:
                    print("Debug: Segmentation results found.")
                return time.time() - start
            time_expired = time.time() - start
        print(f"ERROR: Segmentation was not generated from prediction after waiting {wait_max} seconds.")
        return False

    def upload(self, vtp_fpath, mri_fpath=None, form_participant_id="CC9999", form_session_id=9999, form_gender="Male",
               form_birth_age=30.123, form_birth_weight=0.99, form_singleton=1, form_scan_age=30.123,
               form_scan_number=4, form_radiology_score=2, form_sedation=0):
        start = time.time()
        if self.verbose:
            print("Debug: Accessing Upload from home.")
        bs_home = BeautifulSoup(self.driver.page_source, "lxml")
        assert bs_home.find("p", text="Upload session ID"), "Tried to access Upload from Home but not on Home."
        # Click on Upload
        self.driver.find_element_by_xpath("/html/body/main/div/div[2]/div[2]/div/div[2]/a").click()
        # Fill in the form
        self.driver.find_element_by_name("participant_id").send_keys(form_participant_id)
        self.driver.find_element_by_name("session_id").send_keys(form_session_id)
        self.driver.find_element_by_name("gender").send_keys(form_gender)
        self.driver.find_element_by_name("birth_age").send_keys(str(form_birth_age))
        self.driver.find_element_by_name("birth_weight").send_keys(str(form_birth_weight))
        self.driver.find_element_by_name("singleton").send_keys(str(form_singleton))
        self.driver.find_element_by_name("scan_age").send_keys(str(form_scan_age))
        self.driver.find_element_by_name("scan_number").send_keys(str(form_scan_number))
        self.driver.find_element_by_name("radiology_score").send_keys(str(form_radiology_score))
        self.driver.find_element_by_name("sedation").send_keys(str(form_sedation))
        if mri_fpath:
            if self.verbose:
                print("Debug: Upload of MRI requested.")
            self.driver.find_element_by_name("mri_file").send_keys(os.path.join(os.getcwd(), mri_fpath))
        if vtp_fpath:
            if self.verbose:
                print("Debug: Upload of Surface requested.")
            self.driver.find_element_by_name("surface_file").send_keys(os.path.join(os.getcwd(), vtp_fpath))

        # Click the upload button
        if self.verbose:
            print("Debug: Uploading.")
        bs_upload = BeautifulSoup(self.driver.page_source, "lxml")
        bs_upload_button = bs_upload.find("button", text="Upload")
        self.driver.find_element_by_xpath(self.xpath_soup(bs_upload_button)).click()

        # Retrieve info from website
        bs_results = BeautifulSoup(self.driver.page_source, "lxml")
        bs_table = bs_results.find("table", {"id": "info_table"})
        table_headers = [header.text.strip() for header in bs_table.find_all("th")]
        headers_split = [header.split(" ") for header in table_headers]
        headers = ["_".join(header).lower() for header in headers_split]
        values = [value.text for value in bs_table.find_all("td")]
        dict_table = dict(zip(headers, values))

        # Retrieve passed function arguments
        dict_passed_values = {key[5:]: str(value) for key, value in locals().items() if key[:4] == "form"}

        # Assert that provided info is present
        for key, value in dict_passed_values.items():
            assert key in dict_table, f"Table does not contain {key} which was passed in"
            assert value == dict_table[key], f"Table does contain value {value} passed in, in column {key}"
        return time.time() - start

    def full_lookup_workflow(self, session_id, is_display_MRI=False, is_from_home=True):
        self.home()
        self.login()
        lookup_time = self.lookup(session_id, is_display_MRI, is_from_home)
        predict_time = self.predict()
        segment_time = self.segment()
        if self.verbose:
            print("Debug: Full lookup workflow completed.")
        self.close()
        return lookup_time, predict_time, segment_time

    def full_upload_workflow(self, vtp_fpath, mri_fpath=None, form_participant_id="CC9999",
                             form_session_id=9999, form_gender="Male",
                             form_birth_age=30.123, form_birth_weight=0.99, form_singleton=1, form_scan_age=30.123,
                             form_scan_number=4, form_radiology_score=2, form_sedation=0):
        self.home()
        self.login()
        upload_time = self.upload(vtp_fpath, mri_fpath=mri_fpath, form_participant_id=form_participant_id,
                                  form_session_id=form_session_id, form_gender=form_gender,
                                  form_birth_age=form_birth_age, form_birth_weight=form_birth_weight,
                                  form_singleton=form_singleton,
                                  form_scan_age=form_scan_age, form_scan_number=form_scan_number,
                                  form_radiology_score=form_radiology_score, form_sedation=form_sedation)
        predict_time = self.predict()
        segment_time = self.segment()
        if self.verbose:
            print("Debug: Full upload workflow completed.")
        self.close()
        return upload_time, predict_time, segment_time

    def headless_process(self, session_id):
        self.login()
        self.driver.get(website_url + f"results/{session_id}false")
        predict_time = self.predict()
        segment_time = self.segment()
        self.close()
        return predict_time, segment_time


    def close(self):
        if self.verbose:
            print("Debug: Driver closing.")
        self.driver.close()


tester = WebsiteTester(website_url, login_user, login_pw, verbose=True, headless=False)
# tester.full_upload_workflow("sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp",
#                             mri_fpath="sub-CC00050XX01_ses-7201_T2w_graymatter.nii",
#                             form_session_id=212028)
# tester.full_upload_workflow("sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp",
#                             form_session_id=212031)

tester.full_lookup_workflow(7201, False)
# tester.headless_process(7201)

