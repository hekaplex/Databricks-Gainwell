# Databricks notebook source
class Step:
    def __init__(self, publisher: Publisher, number: int, instructions: str, after_codebox_instructions: str, add_catalog_input: bool, add_username_input: bool, width: int, test_code: str = None, label: str = None, expected: str = None, length: int = 0):

        assert type(publisher) == Publisher, f"Expected the parameter \"da\" to be of type DBAcademyHelper, found {type(publisher)}"
        assert type(number) == int, f"Expected the parameter \"number\" to be of type int, found {type(number)}"
        assert type(width) == int, f"Expected the parameter \"width\" to be of type int, found {type(width)}"
        assert instructions is None or type(instructions) == str, f"Expected the parameter \"instructions\" to be of type str, found {type(instructions)}"
        assert after_codebox_instructions is None or type(after_codebox_instructions) == str, f"Expected the parameter \"after_codebox_instructions\" to be of type str, found {type(after_codebox_instructions)}"
        assert type(add_catalog_input) == bool, f"Expected the parameter \"add_catalog_input\" to be of type bool, found {type(add_catalog_input)}"
        assert type(add_username_input) == bool, f"Expected the parameter \"add_username_input\" to be of type bool, found {type(add_username_input)}"
        assert test_code is None or type(test_code) == str, f"Expected the parameter \"test_code\" to be of type str, found {type(test_code)}"
        assert label is None or type(label) == str, f"Expected the parameter \"label\" to be of type str, found {type(label)}"
        assert expected is None or type(expected) == str, f"Expected the parameter \"expected\" to be of type str, found {type(expected)}"
        assert type(length) == int, f"Expected the parameter \"length\" to be of type int, found {type(length)}"

        self.publisher = publisher
        self.da = publisher.da
        self.width = width
        
        self.number = number
        self.id = f"step-{number}"
        self.instructions = instructions
        self.after_codebox_instructions = after_codebox_instructions
        self.add_catalog_input = add_catalog_input
        self.add_username_input = add_username_input
        
        self.statements = []

        self.test_code = test_code.strip() if test_code is not None else ""
        self.label = label
        self.expected = expected
        self.length = length

    def add(self, statement):
        assert statement is not None, "The statement must be specified"
        statement = statement.strip()
        if statement != "":
            assert statement.endswith(";"), f"Expected statement to end with a semi-colon:\n{statement}"
            assert statement.count(";") == 1, f"Found more than one statement in the specified SQL string:\n{statement}"
        self.statements.append(statement)
    
    def update_statement(self, statement):
        statement = statement.replace("{catalog_name}", self.da.catalog_name)
        statement = statement.replace("{username}", self.da.username)

        self.validate_statement(statement)
        return statement.strip()

    @staticmethod
    def validate_statement(statement):
        import re

        test = re.search(r"{{\s*[\w.]*\s*}}", statement)
        if test is not None:
            raise Exception(f"Found existing mustache template: {test}")

        test = re.search(r"{\s*[\w.]*\s*}", statement)
        if test is not None:
            raise Exception(f"Found existing mustache template: {test}")

    def render(self, debug=False):
        if dbgems.is_generating_docs():
            print(f"Rendering suppressed; publishing")
            return None
        else:
            html = self.to_html(render_alone=True, debug=debug)
            dbgems.display_html(html)
            return html
        
    def execute(self):
        if dbgems.is_generating_docs():
            print(f"Execution suppressed; publishing")
        else:
            for statement in self.statements:
                if len(statement) == 0:
                    print(f"")
                else:
                    statement = self.update_statement(statement)
                    print(f"- Executing " + ("-"*68))
                    print(statement)
                    print("-"*80)
                    results = dbgems.sql(statement)
                    dbgems.display(results)

    def to_html(self, render_alone=False, debug=False):
        sql = ""
        for i, statement in enumerate(self.statements):
            sql += statement.strip()
            sql += "\n"
        sql = sql.strip()
        sql_row_count = len(sql.split("\n"))+1

        html = ""

        if render_alone:
            # This is a partial render, so we need body and style tags
            html = f"""
                <!DOCTYPE html>
                <html lang="en">
                    <head>
                        <meta charset="utf-8"/>
                        <link href='https://fonts.googleapis.com/css?family=DM Sans' rel='stylesheet'>
                        {self.publisher.common_style}
                        <script>
                            {self.publisher.answer_is}
                        </script>
                    </head>
                    <body>
                        <div style="border-style: outset; padding: 5px; width:{self.width+5}">"""

        html += f"""<div id="{self.id}-wrapper" style="width:{self.width}px;">"""

        if self.instructions is not None:
            instructions = self.instructions
            if "{step}" in instructions: instructions = instructions.format(step=self.number)

            html += f"""<div id="{self.id}-instruction" style="margin-bottom:1em">{instructions}</div>"""

        # This is the validation logic
        display_test_code = "display: none;" if self.test_code == "" else ""
        display_button = "display:none;" if sql == "" else ""
        test_row_count = len(self.test_code.split("\n")) + 1
        html += f"""
        <div style="{display_test_code}">
            <div>
                <textarea id="{self.id}-test-ta" style="width:100%; padding:10px; overflow-x:auto" rows="{test_row_count}">{self.test_code}</textarea>
                <textarea id="{self.id}-test-backup" style="display:none;">{self.test_code}</textarea>
            </div>
            <div style="text-align:right; {display_test_code} margin-top:5px">
                <button id="{self.id}-test-show-btn" type="button" style="{display_button}" 
                        onclick="document.getElementById('{self.id}-sql').style.display = 'block';">Show Answer</button>
                <button id="{self.id}-test-btn" type="button"  onclick="
                    let ta = document.getElementById('{self.id}-test-ta');
                    ta.select();
                    ta.setSelectionRange(0, ta.value.length);
                    navigator.clipboard.writeText(ta.value);">Copy</button>
            </div>
        </div>
          """
        if self.label is not None:
            html+= f"""
        <div>
            <table style="margin:1em 0; border-collapse:collapse; width:{self.width}px;">
                <tbody>
                    <tr>
                        <td style="background-color: #D1E2FF; width: 100%; text-align:left;">{self.label}</td>
                        <td style="background-color: #D1E2FF; width: 1em; text-align:right;">
                            <input type="text" size="{self.length}" onchange="answerIs(this, ['{self.expected}']);" style="background-color: rgb(255, 255, 255);">
                        </td>
                    </tr>    
                </tbody>
            </table>
        </div>
        """
        
        # This is the "standard" display for SQL content
        if display_test_code == "" or sql == "":
            # We are showing test code so don't display SQL or there is no SQL
            display_sql = "display: none;"
        else:
            display_sql = ""

        html += f"""
        <div id="{self.id}-sql" style="width:{self.width}px; {display_sql}">
            <div>
                <textarea id="{self.id}-sql-ta" style="width:100%; padding:10px" rows="{sql_row_count}">{sql}</textarea>
                <textarea id="{self.id}-sql-backup" style="display:none;">{sql}</textarea>
            </div>
            <div style="text-align:right; margin-top:5px">
                <button id="{self.id}-sql-btn" type="button"  onclick="
                    let ta = document.getElementById('{self.id}-sql-ta');
                    ta.select();
                    ta.setSelectionRange(0, ta.value.length);
                    navigator.clipboard.writeText(ta.value);">Copy</button>
            </div>
        </div>
        """
        
        if self.after_codebox_instructions is not None:
            after_codebox_instructions = self.after_codebox_instructions
            html += f"""<div id="{self.id}-after_codebox_instruction" style="margin-bottom:1em">{after_codebox_instructions}</div>"""
            
        if self.add_catalog_input:
            html += """<table>
                        <tr>
                            <td style="white-space:nowrap">Catalog Name:&nbsp;</td>
                            <td><input id="catalog_name" type="text" style="width:40em" onchange="update();"></td>
                        </tr>
                        <tr>
                            <td>Current Catalog Name:</td>
                            <td id="catalog-input-msg"></td>
                        </tr>
                    </table>"""
        
        if self.add_username_input:
            html += """<table>
                        <tr>
                            <td style="white-space:nowrap">Username:&nbsp;</td>
                            <td><input id="username" type="text" style="width:40em" onchange="update();"></td>
                        </tr>
                        <tr>
                            <td>Current Username:</td>
                            <td id="username-input-msg"></td>
                        </tr>
                    </table>"""
            
        html += "</div>"
            
        if render_alone:
            # If the style was specified,
            # we need to provide closing tags
            html += "</div></body></html>"

        if debug:
            file_name = self.publisher.file_name.replace(".html", f"_{self.number+1}.html")
            dbgems.get_dbutils().fs.put(f"dbfs:/FileStore/{file_name}", contents=html, overwrite=True)
            dbgems.display_html(f"""<a href="/files/{file_name}">download</a>""")

        return html

