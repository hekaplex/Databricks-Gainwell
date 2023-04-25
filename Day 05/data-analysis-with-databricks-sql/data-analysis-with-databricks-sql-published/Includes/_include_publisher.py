# Databricks notebook source
class Publisher:
    def __init__(self, da: DBAcademyHelper):
        import re

        self.da = da
        self.steps = list()
        self.width = 800

        self.lesson_name = dbgems.get_notebook_name()
        self.file_name = re.sub(r"[^a-zA-Z\d]", "-", self.lesson_name)+".html"
        while "--" in self.file_name: self.file_name = self.file_name.replace("--", "-")

        self.answer_is = """
            function answerIs(self, expected) {
                if (self.value === "") {
                    self.style.backgroundColor="#ffffff";
                } else if (self.value.toLowerCase().includes(expected)) {
                    self.style.backgroundColor="#7ffe78";
                } else {
                    self.style.backgroundColor="#ffb9bb";
                }
            }
        """

        self.common_style = f"""
            <style>
                body {{background-color: #f9f7f4;font-family: 'DM Sans', serif;}}
                h2 {{color: #ff3621;}}
                h3 {{margin-left: 5px}}
                ol {{margin-left: 20px; font-family:sans-serif; color: #0873FF;}} //E81334 618794
                ol li {{margin-bottom:5px;}}
                td {{padding: 5px;border-bottom: 1px solid #ededed;}}
                tr.selected td {{color: white;background-color: red;}}
                .monofont {{font-family: monospace; font-size: 14px}}
                .content {{max-width: 800px; margin: auto; padding-left: 50px}}
                .image-icon-inline {{display:inline; vertical-align: middle; margin: 0 10px 0 10px}}
                .instructions-div {{padding-left: 5px}}
            </style>""".strip()
    
    def add_step(self, include_use, *, instructions, after_codebox_instructions="", statements=None, add_catalog_input=False, add_username_input=False):
        return self.add(include_use=include_use,
                        instructions=instructions,
                        after_codebox_instructions=after_codebox_instructions,
                        statements=statements,
                        add_catalog_input=add_catalog_input,
                        add_username_input=add_username_input,                        
                        test_code=None,
                        label=None,
                        expected=None,
                        length=1)

    def add_validation(self, include_use, *, instructions, statements, test_code, label, expected, length, after_codebox_instructions="", add_catalog_input=False, add_username_input=False):
        return self.add(include_use=include_use,
                        instructions=instructions,
                        after_codebox_instructions=after_codebox_instructions,
                        statements=statements,
                        add_catalog_input=add_catalog_input,
                        add_username_input=add_username_input,                        
                        test_code=test_code,
                        label=label,
                        expected=expected,
                        length=length)

    def add(self, *, include_use, instructions, after_codebox_instructions, statements, add_catalog_input, add_username_input, test_code, label, expected, length):
        # Convert single statement to the required list
        if statements is None: statements = []
        if type(statements) == str: statements = [statements]
        
        step = Step(publisher=self,
                    number=len(self.steps)+1,
                    instructions=instructions,
                    after_codebox_instructions=after_codebox_instructions,
                    add_catalog_input=add_catalog_input,
                    add_username_input=add_username_input,                        
                    width=self.width,
                    test_code=test_code,
                    label=label,
                    expected=expected,
                    length=length)

        if include_use:
            step.add("USE {catalog_name};")

        for statement in statements:
            step.add(statement)
            
        self.steps.append(step)
        return step
        
    def find_repo(self):

        for i in range(1, 10):
            repo_path = dbgems.get_notebook_dir(offset=-i)
            status = self.da.client.workspace().get_status(repo_path)
            if status.get("object_type") == "REPO":
                return status

        raise Exception(f"The repo directory was not found")

    def publish(self, include_inputs=True, inputs_inline=False):
        import os

        if dbgems.get_cloud() != "AWS":
            message = f"Skipping publish while running on {dbgems.get_cloud()}"
            print(message)
            return f"<html><body><p>{message}</p></body></html>"
        
        html = self.to_html(self.lesson_name, include_inputs, inputs_inline)

        repo_path = self.find_repo().get("path")
        target_file = f"/Workspace{repo_path}/docs/{self.file_name}"
        print(f"Target:  {target_file}")
        
        target_dir = "/".join(target_file.split("/")[:-1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        with open(target_file, "w") as w:
            w.write(html)
        
        url = f"https://urban-umbrella-8e9e1ba3.pages.github.io/{self.file_name}"
        dbgems.display_html(f"<a href='{url}' target='_blank'>GitHub Page</a> (Changes must be committed before seeing the new page)")

        return html
        
    def to_html(self, lesson_name, include_inputs, inputs_inline):
        from datetime import date
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>{lesson_name}</title>
                <link href='https://fonts.googleapis.com/css?family=DM%20Sans' rel='stylesheet' type='text/css'>
                {self.common_style}
            </head>
            <body onload="loaded()">
                <div class="content">
                    <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/db-academy-rgb-1200px_no_bg.png" 
                        alt="Databricks Learning" 
                        style="width: 600px; margin-left: 100px; margin-right: 100px">
                <hr>
                <h1>{lesson_name}</h1>"""
        if include_inputs: 
            html += """<div id="inputs">
                    <p>The two fields below are used to customize queries used in this course.</p>
                    <p><span style="color:red">Note:</span> If the two fields below do not contain your catalog name and username, you either do not have cookies enabled, or you need to follow the instructions under "The Query Editor" <a href="Demo-A-Simple-but-Quick-Query-Visualization-and-Dashboard.html" target="_blank">here</a>.</p>
                    <table>
                        <tr>
                            <td style="white-space:nowrap">Catalog Name:&nbsp;</td>
                            <td><input id="catalog_name" type="text" style="width:40em" onchange="update();"></td>
                        </tr><tr>
                            <td style="white-space:nowrap">Username:&nbsp;</td>
                            <td><input id="username" type="text" style="width:40em" onchange="update();"></td>
                        </tr>
                    </table>
                </div>"""
        
        for step in self.steps:
            html += "<hr>\n"
            html += step.to_html()
        
        # The ID elements that must be updated at runtime
        ids = [s.id for s in self.steps]  
        
        # F-strings complicate how to compose the following block of code so we inject the
        # following tokens to skirt various escaping issues related to f-strings
        username_token = "{username}"          # The runtime token for the username
        catalog_name_token = "{catalog_name}"  # The runtime token for the catalog_name
        
        # Create the Javascript used to update the HTMl at runtime
        html += f"""<script>
            {self.answer_is}
            function loaded() {{
                let data = document.cookie;
                if (data != null && data.trim() !== "") {{
                    let parts = data.split(";");
                    for (let i = 0; i < parts.length; i++) {{
                        let key_value = parts[i].trim();
                        let key = key_value.split("=")[0].trim();
                        let value = key_value.split("=")[1].trim();
                        if (value !== "n/a") {{
                            if (key === "{self.da.course_config.course_code}_catalog_name") {{
                                document.getElementById("catalog_name").value = value;
                            }} else if (key === "{self.da.course_config.course_code}_username") {{
                                document.getElementById("username").value = value;
                            }} else {{
                                console.log("Unknown cookie: "+key);
                            }}
                        }}
                    }}
                }}
                update();
            }}
            function update() {{      
                let catalog_name = document.getElementById("catalog_name").value;
                let username = document.getElementById("username").value;
                let ids = {ids};

                if (catalog_name === null || username === null || 
                    catalog_name === "" || username === "" || 
                    catalog_name == "n/a" || username === "n/a") {{
                    for (let i = 0; i < ids.length; i++) {{
                        document.getElementById(ids[i]+"-test-ta").disabled = true;
                        document.getElementById(ids[i]+"-sql-ta").disabled = true;

                        document.getElementById(ids[i]+"-test-btn").disabled = true;
                        document.getElementById(ids[i]+"-sql-btn").disabled = true;

                        document.getElementById(ids[i]+"-test-ta").value = document.getElementById(ids[i]+"-test-backup").value
                        document.getElementById(ids[i]+"-sql-ta").value = document.getElementById(ids[i]+"-sql-backup").value
                    }}
                }} else if (catalog_name == "n/a" || username === "n/a") {{
                    for (let i = 0; i < ids.length; i++) {{
                        document.getElementById(ids[i]+"-test-ta").disabled = false;
                        document.getElementById(ids[i]+"-sql-ta").disabled = false;

                        document.getElementById(ids[i]+"-test-btn").disabled = false;
                        document.getElementById(ids[i]+"-sql-btn").disabled = false;
                    }}    
                }} else {{
                    let illegals = ["<",">","\\"","'","\\\\","/"]; 
                    for (let i= 0; i < illegals.length; i++) {{
                        if (catalog_name.includes(illegals[i])) {{
                            alert("Please double check your catalog name.\\nIt cannot not include the " + illegals[i] + " symbol.");
                            return;
                        }}
                        if (username.includes(illegals[i])) {{
                            alert("Please double check your username.\\nIt cannot not include the " + illegals[i] + " symbol.");
                            return;
                        }}
                    }}
                    if (catalog_name.includes("@")) {{
                        alert("Please double check your catalog name.\\nIt should not include the @ symbol.");
                        return;
                    }}
                    if (username.includes("@") === false) {{
                        alert("Please double check your username.\\nIt should include the @ symbol.");
                        return;
                    }}
                    """
        if(inputs_inline):
            html += f"""
                    document.getElementById("catalog-input-msg").innerText = catalog_name;
                    document.getElementById("username-input-msg").innerText = username;
                    """
        html += f"""
                    for (let i = 0; i < ids.length; i++) {{
                        document.getElementById(ids[i]+"-test-ta").disabled = false;
                        document.getElementById(ids[i]+"-sql-ta").disabled = false;

                        document.getElementById(ids[i]+"-test-btn").disabled = false;
                        document.getElementById(ids[i]+"-sql-btn").disabled = false;

                        document.getElementById(ids[i]+"-test-ta").value = document.getElementById(ids[i]+"-test-backup").value
                                                                                   .replaceAll("{catalog_name_token}", catalog_name)
                                                                                   .replaceAll("{username_token}", username);

                        document.getElementById(ids[i]+"-sql-ta").value = document.getElementById(ids[i]+"-sql-backup").value
                                                                                  .replaceAll("{catalog_name_token}", catalog_name)
                                                                                  .replaceAll("{username_token}", username);

                        document.cookie = "{self.da.course_config.course_code}_catalog_name="+catalog_name;
                        document.cookie = "{self.da.course_config.course_code}_username="+username;
                    }}
                }}
            }}
        </script>"""
        
        if not include_inputs:
            html += f"""<script>
                    document.getElementById("catalog_name").value = "n/a";
                    document.getElementById("username").value = "n/a";
                    
            </script>"""
        
        if inputs_inline:
            html += f"""<script>
                    document.getElementById("catalog_name").value = "None";
                    document.getElementById("username").value = "Input username (including the @ symbol)";
            </script>"""
        
        version = dbgems.get_parameter("version", "N/A")
        
        html += f"""
        <hr style="margin-top:8em">
        <div>
            <div>&copy; {date.today().year} Databricks, Inc. All rights reserved.</div>
            <div>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.</div>
            <div style="margin-top:1em">
                <div style="float:left">
                    <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
                    <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
                </div>
                <div style="float:right">{version}</div>
            </div>               
        </div>"""
            
        html += """</div></body></html>"""
        return html

