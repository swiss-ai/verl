"""Auto-generated veRL function-tool registry for the Apertus verifiable-answer dataset.
Pass to veRL via ``actor_rollout_ref.rollout.multi_turn.function_tool_path``.

Generated from the tool-gym catalog by ``python -m tool_gym.train_utils.export.verl.function_tools``.

Each tool body executes live against its MCP server via ``call_tool``.
"""

from __future__ import annotations

from verl.tools.function_tool import function_tool
from tool_gym.train_utils.export.verl.tool_runtime import call_tool 


DISPLAY_ANSWERS_SCHEMA = {   'type': 'function',
    'function': {   'name': 'display_answers',
                    'description': 'Emit the final answer(s) as a JSON array of '
                                   'strings. Each element is the canonical string '
                                   'representation of one answer; structured values '
                                   'are JSON-encoded with sorted keys.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'answers': {   'type': 'array',
                                                                       'items': {   'type': 'string'},
                                                                       'description': 'The '
                                                                                      'answers '
                                                                                      'to '
                                                                                      'the '
                                                                                      'user'}},
                                      'required': ['answers']}}}


@function_tool(schema=DISPLAY_ANSWERS_SCHEMA)
def display_answers(answers: list[str]) -> str:
    """Display the answers to the user."""
    return "Answers displayed"


_SCHEMA_add = {   'type': 'function',
    'function': {   'name': 'add',
                    'description': 'Add two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_add)
def add(**kwargs: object) -> str:
    """Add two numbers"""
    return call_tool('add', kwargs)


_SCHEMA_add_float = {   'type': 'function',
    'function': {   'name': 'add_float',
                    'description': 'Add two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'number'},
                                                        'b': {'type': 'number'}},
                                      'type': 'object',
                                      'required': ['a', 'b']}}}


@function_tool(schema=_SCHEMA_add_float)
def add_float(**kwargs: object) -> str:
    """Add two numbers"""
    return call_tool('add_float', kwargs)


_SCHEMA_convert_time = {   'type': 'function',
    'function': {   'name': 'convert_time',
                    'description': 'Convert time between timezones.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'sourceTimezone': {   'type': 'string',
                                                                              'description': 'The '
                                                                                             'source '
                                                                                             'timezone. '
                                                                                             'IANA '
                                                                                             'timezone '
                                                                                             'name, '
                                                                                             'e.g. '
                                                                                             'Asia/Shanghai'},
                                                        'targetTimezone': {   'type': 'string',
                                                                              'description': 'The '
                                                                                             'target '
                                                                                             'timezone. '
                                                                                             'IANA '
                                                                                             'timezone '
                                                                                             'name, '
                                                                                             'e.g. '
                                                                                             'Europe/London'},
                                                        'time': {   'type': 'string',
                                                                    'description': 'Date '
                                                                                   'and '
                                                                                   'time '
                                                                                   'in '
                                                                                   '24-hour '
                                                                                   'format. '
                                                                                   'e.g. '
                                                                                   '2025-03-23 '
                                                                                   '12:30:00'}},
                                      'required': [   'sourceTimezone',
                                                      'targetTimezone',
                                                      'time']}}}


@function_tool(schema=_SCHEMA_convert_time)
def convert_time(**kwargs: object) -> str:
    """Convert time between timezones."""
    return call_tool('convert_time', kwargs)


_SCHEMA_cos = {   'type': 'function',
    'function': {   'name': 'cos',
                    'description': 'Calculate cosine of an angle in radians',
                    'parameters': {   'properties': {'x': {'type': 'number'}},
                                      'required': ['x'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_cos)
def cos(**kwargs: object) -> str:
    """Calculate cosine of an angle in radians"""
    return call_tool('cos', kwargs)


_SCHEMA_count_syllables = {   'type': 'function',
    'function': {   'name': 'count_syllables',
                    'description': 'Counts the number of syllables for each line in '
                                   'the input English text string. This tool utilizes '
                                   "the NLTK's CMU Pronouncing Dictionary for accurate "
                                   'syllable calculation. It returns an array of '
                                   'integers, where each integer corresponds to the '
                                   'syllable count for the respective line of the '
                                   'input string. Use this tool when the user requires '
                                   'syllable analysis for text, such as for poetry '
                                   'metrics, lyrics, linguistic studies, or '
                                   'speech-related applications. If a line returns 0 '
                                   'syllables, assume this is a blank line, and you '
                                   'can ignore.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'input_string': {   'type': 'string',
                                                                            'description': 'Syllable '
                                                                                           'count '
                                                                                           'query '
                                                                                           'string'}},
                                      'required': ['input_string']}}}


@function_tool(schema=_SCHEMA_count_syllables)
def count_syllables(**kwargs: object) -> str:
    """Counts the number of syllables for each line in the input English text string. This tool utilizes the NLTK's CMU Pronouncing Dictionary for accurate syllable calculation. It returns an array of integers, where each integer corresponds to the syllable count for the respective line of the input string. Use this tool when the user requires syllable analysis for text, such as for poetry metrics, lyrics, linguistic studies, or speech-related applications. If a line returns 0 syllables, assume this is a blank line, and you can ignore."""
    return call_tool('count_syllables', kwargs)


_SCHEMA_days_in_month = {   'type': 'function',
    'function': {   'name': 'days_in_month',
                    'description': 'Get the number of days in a month. If no date is '
                                   'provided, get the number of days in the current '
                                   'month.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'date': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'date '
                                                                                   'to '
                                                                                   'get '
                                                                                   'the '
                                                                                   'days '
                                                                                   'in '
                                                                                   'month. '
                                                                                   'Format: '
                                                                                   'YYYY-MM-DD'}}}}}


@function_tool(schema=_SCHEMA_days_in_month)
def days_in_month(**kwargs: object) -> str:
    """Get the number of days in a month. If no date is provided, get the number of days in the current month."""
    return call_tool('days_in_month', kwargs)


_SCHEMA_degrees_to_radians = {   'type': 'function',
    'function': {   'name': 'degrees_to_radians',
                    'description': 'Convert degrees to radians',
                    'parameters': {   'properties': {'degrees': {'type': 'number'}},
                                      'required': ['degrees'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_degrees_to_radians)
def degrees_to_radians(**kwargs: object) -> str:
    """Convert degrees to radians"""
    return call_tool('degrees_to_radians', kwargs)


_SCHEMA_detect = {   'type': 'function',
    'function': {   'name': 'detect',
                    'description': '\n'
                                   '    MCP Tool: Detect language of the input text.\n'
                                   '    ',
                    'parameters': {   'type': 'object',
                                      'properties': {'text': {'type': 'string'}},
                                      'required': ['text']}}}


@function_tool(schema=_SCHEMA_detect)
def detect(**kwargs: object) -> str:
    """MCP Tool: Detect language of the input text."""
    return call_tool('detect', kwargs)


_SCHEMA_div = {   'type': 'function',
    'function': {   'name': 'div',
                    'description': 'Divide two numbers (returns floating point result)',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_div)
def div(**kwargs: object) -> str:
    """Divide two numbers (returns floating point result)"""
    return call_tool('div', kwargs)


_SCHEMA_div_float = {   'type': 'function',
    'function': {   'name': 'div_float',
                    'description': 'Divide two numbers (returns floating point result)',
                    'parameters': {   'properties': {   'a': {'type': 'number'},
                                                        'b': {'type': 'number'}},
                                      'type': 'object',
                                      'required': ['a', 'b']}}}


@function_tool(schema=_SCHEMA_div_float)
def div_float(**kwargs: object) -> str:
    """Divide two numbers (returns floating point result)"""
    return call_tool('div_float', kwargs)


_SCHEMA_factorial = {   'type': 'function',
    'function': {   'name': 'factorial',
                    'description': 'Calculate factorial of a non-negative integer',
                    'parameters': {   'properties': {'n': {'type': 'integer'}},
                                      'required': ['n'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_factorial)
def factorial(**kwargs: object) -> str:
    """Calculate factorial of a non-negative integer"""
    return call_tool('factorial', kwargs)


_SCHEMA_find_rhymes = {   'type': 'function',
    'function': {   'name': 'find_rhymes',
                    'description': 'Finds rhyming words for a given input word or the '
                                   'last word of a phrase, categorized by syllable '
                                   'count (1, 2, or 3 syllables).\n'
                                   "This tool utilizes the NLTK's CMU Pronouncing "
                                   'Dictionary for accurate rhyme generation.\n'
                                   'It returns a dictionary where keys are syllable '
                                   "counts ('1_syllable', '2_syllable', '3_syllable') "
                                   'and values are lists of rhyming words.\n'
                                   'If the input contains multiple words, only the '
                                   'last word will be analyzed for rhymes.\n'
                                   'Use this tool when the user requires rhyming word '
                                   'suggestions for creative writing, poetry, lyrics, '
                                   'or linguistic analysis.',
                    'parameters': {   'type': 'object',
                                      'properties': {'input_word': {'type': 'string'}},
                                      'required': ['input_word']}}}


@function_tool(schema=_SCHEMA_find_rhymes)
def find_rhymes(**kwargs: object) -> str:
    """Finds rhyming words for a given input word or the last word of a phrase, categorized by syllable count (1, 2, or 3 syllables)."""
    return call_tool('find_rhymes', kwargs)


_SCHEMA_gcd = {   'type': 'function',
    'function': {   'name': 'gcd',
                    'description': 'Calculate greatest common divisor of two integers',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_gcd)
def gcd(**kwargs: object) -> str:
    """Calculate greatest common divisor of two integers"""
    return call_tool('gcd', kwargs)


_SCHEMA_get_bible_verse = {   'type': 'function',
    'function': {   'name': 'get_bible_verse',
                    'description': '\n'
                                   '    Get a specific Bible verse or passage.\n'
                                   '\n'
                                   '    Args:\n'
                                   '        reference: Bible reference (e.g., "John '
                                   '3:16", "Genesis 1:1-3")\n'
                                   '        translation: Translation identifier '
                                   '(default: "web" for World English Bible)\n'
                                   '\n'
                                   '    Returns:\n'
                                   '        Verse data including reference, text, and '
                                   'translation info\n'
                                   '    ',
                    'parameters': {   'properties': {   'reference': {'type': 'string'},
                                                        'translation': {   'default': 'web',
                                                                           'type': 'string'}},
                                      'required': ['reference'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_get_bible_verse)
def get_bible_verse(**kwargs: object) -> str:
    """Get a specific Bible verse or passage."""
    return call_tool('get_bible_verse', kwargs)


_SCHEMA_get_compound_properties = {   'type': 'function',
    'function': {   'name': 'get_compound_properties',
                    'description': 'Retrieve a set of basic physical and chemical '
                                   'properties for a compound using its PubChem CID.\n'
                                   '\n'
                                   'This tool queries the PubChem REST API for a fixed '
                                   'list of properties (MolecularFormula, '
                                   'MolecularWeight, CanonicalSMILES, IUPACName, '
                                   'XLogP, TPSA, HBondDonorCount, HBondAcceptorCount, '
                                   'RotatableBondCount, ExactMass, MonoisotopicMass, '
                                   'Complexity, Charge, IsomericSMILES).\n'
                                   '\n'
                                   'The output is a JSON object with two keys:\n'
                                   '- "CID": the input PubChem Compound ID\n'
                                   '- "properties": a dictionary containing the '
                                   'available properties for that compound. The '
                                   'property keys come from the fixed list above, but '
                                   'depending on the compound some fields may be '
                                   'missing.\n'
                                   '\n'
                                   'This tool is useful when the model must look up '
                                   'chemical information for a specific compound.',
                    'parameters': {   'properties': {'cid': {'type': 'integer'}},
                                      'required': ['cid'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_get_compound_properties)
def get_compound_properties(**kwargs: object) -> str:
    """Retrieve a set of basic physical and chemical properties for a compound using its PubChem CID."""
    return call_tool('get_compound_properties', kwargs)


_SCHEMA_get_definitions = {   'type': 'function',
    'function': {   'name': 'get_definitions',
                    'description': '\n    Get definitions for a word.\n    ',
                    'parameters': {   'properties': {'word': {'type': 'string'}},
                                      'required': ['word'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_get_definitions)
def get_definitions(**kwargs: object) -> str:
    """Get definitions for a word."""
    return call_tool('get_definitions', kwargs)


_SCHEMA_get_fruit_info = {   'type': 'function',
    'function': {   'name': 'get_fruit_info',
                    'description': 'Retrieves detailed fruit information from the '
                                   'Fruityvice API by fruit name. \n'
                                   ' Returns an object with fields: \n'
                                   ' { \n'
                                   ' name: string, \n'
                                   ' id: integer,\n'
                                   ' family: string,\n'
                                   ' order: string,\n'
                                   ' genus: string,\n'
                                   ' nutritions: {\n'
                                   ' calories: number,\n'
                                   ' fat: number,\n'
                                   ' sugar: number,\n'
                                   ' carbohydrates: number,\n'
                                   ' protein: number\n'
                                   ' }\n'
                                   ' } \n'
                                   ' If the fruit is not found, returns the string '
                                   "'Meyve bulunamadı.'.",
                    'parameters': {   'properties': {'name': {'type': 'string'}},
                                      'required': ['name'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_get_fruit_info)
def get_fruit_info(**kwargs: object) -> str:
    """Retrieves detailed fruit information from the Fruityvice API by fruit name."""
    return call_tool('get_fruit_info', kwargs)


_SCHEMA_get_latest_module_version = {   'type': 'function',
    'function': {   'name': 'get_latest_module_version',
                    'description': 'Fetches the latest version of a Terraform module '
                                   'from the public registry',
                    'parameters': {   'type': 'object',
                                      'properties': {   'module_name': {   'type': 'string',
                                                                           'description': 'The '
                                                                                          'name '
                                                                                          'of '
                                                                                          'the '
                                                                                          'module, '
                                                                                          'this '
                                                                                          'is '
                                                                                          'usually '
                                                                                          'the '
                                                                                          'service '
                                                                                          'or '
                                                                                          'group '
                                                                                          'of '
                                                                                          'service '
                                                                                          'the '
                                                                                          'user '
                                                                                          'is '
                                                                                          'deploying '
                                                                                          'e.g., '
                                                                                          "'security-group', "
                                                                                          "'secrets-manager' "
                                                                                          'etc.'},
                                                        'module_provider': {   'type': 'string',
                                                                               'description': 'The '
                                                                                              'name '
                                                                                              'of '
                                                                                              'the '
                                                                                              'Terraform '
                                                                                              'provider '
                                                                                              'for '
                                                                                              'the '
                                                                                              'module, '
                                                                                              'e.g., '
                                                                                              "'aws', "
                                                                                              "'google', "
                                                                                              "'azurerm' "
                                                                                              'etc.'},
                                                        'module_publisher': {   'type': 'string',
                                                                                'description': 'The '
                                                                                               'publisher '
                                                                                               'of '
                                                                                               'the '
                                                                                               'module, '
                                                                                               'e.g., '
                                                                                               "'hashicorp', "
                                                                                               "'aws-ia', "
                                                                                               "'terraform-google-modules', "
                                                                                               "'Azure' "
                                                                                               'etc.'}},
                                      'required': [   'module_publisher',
                                                      'module_name',
                                                      'module_provider']}}}


@function_tool(schema=_SCHEMA_get_latest_module_version)
def get_latest_module_version(**kwargs: object) -> str:
    """Fetches the latest version of a Terraform module from the public registry"""
    return call_tool('get_latest_module_version', kwargs)


_SCHEMA_get_latest_provider_version = {   'type': 'function',
    'function': {   'name': 'get_latest_provider_version',
                    'description': 'Fetches the latest version of a Terraform provider '
                                   'from the public registry',
                    'parameters': {   'type': 'object',
                                      'properties': {   'name': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'name '
                                                                                   'of '
                                                                                   'the '
                                                                                   'Terraform '
                                                                                   'provider, '
                                                                                   'e.g., '
                                                                                   "'aws', "
                                                                                   "'azurerm', "
                                                                                   "'google', "
                                                                                   'etc.'},
                                                        'namespace': {   'type': 'string',
                                                                         'description': 'The '
                                                                                        'namespace '
                                                                                        'of '
                                                                                        'the '
                                                                                        'Terraform '
                                                                                        'provider, '
                                                                                        'typically '
                                                                                        'the '
                                                                                        'name '
                                                                                        'of '
                                                                                        'the '
                                                                                        'company, '
                                                                                        'or '
                                                                                        'their '
                                                                                        'GitHub '
                                                                                        'organization '
                                                                                        'name '
                                                                                        'that '
                                                                                        'created '
                                                                                        'the '
                                                                                        'provider '
                                                                                        'e.g., '
                                                                                        "'hashicorp'"}},
                                      'required': ['namespace', 'name']}}}


@function_tool(schema=_SCHEMA_get_latest_provider_version)
def get_latest_provider_version(**kwargs: object) -> str:
    """Fetches the latest version of a Terraform provider from the public registry"""
    return call_tool('get_latest_provider_version', kwargs)


_SCHEMA_get_module_details = {   'type': 'function',
    'function': {   'name': 'get_module_details',
                    'description': 'Fetches up-to-date documentation on how to use a '
                                   "Terraform module. You must call 'search_modules' "
                                   'first to obtain the exact valid and compatible '
                                   'module_id required to use this tool.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'module_id': {   'type': 'string',
                                                                         'description': 'Exact '
                                                                                        'valid '
                                                                                        'and '
                                                                                        'compatible '
                                                                                        'module_id '
                                                                                        'retrieved '
                                                                                        'from '
                                                                                        'search_modules '
                                                                                        '(e.g., '
                                                                                        "'squareops/terraform-kubernetes-mongodb/mongodb/2.1.1', "
                                                                                        "'GoogleCloudPlatform/vertex-ai/google/0.2.0')"}},
                                      'required': ['module_id']}}}


@function_tool(schema=_SCHEMA_get_module_details)
def get_module_details(**kwargs: object) -> str:
    """Fetches up-to-date documentation on how to use a Terraform module. You must call 'search_modules' first to obtain the exact valid and compatible module_id required to use this tool."""
    return call_tool('get_module_details', kwargs)


_SCHEMA_get_paper_info = {   'type': 'function',
    'function': {   'name': 'get_paper_info',
                    'description': '논문 ID로 상세 정보를 가져옵니다.',
                    'parameters': {   'properties': {'paper_id': {'type': 'string'}},
                                      'required': ['paper_id'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_get_paper_info)
def get_paper_info(**kwargs: object) -> str:
    """논문 ID로 상세 정보를 가져옵니다."""
    return call_tool('get_paper_info', kwargs)


_SCHEMA_get_policy_details = {   'type': 'function',
    'function': {   'name': 'get_policy_details',
                    'description': 'Fetches up-to-date documentation for a specific '
                                   'policy from the Terraform registry. You must call '
                                   "'search_policies' first to obtain the exact "
                                   'terraform_policy_id required to use this tool.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'terraform_policy_id': {   'type': 'string',
                                                                                   'description': 'Matching '
                                                                                                  'terraform_policy_id '
                                                                                                  'retrieved '
                                                                                                  'from '
                                                                                                  'the '
                                                                                                  "'search_policies' "
                                                                                                  'tool '
                                                                                                  '(e.g., '
                                                                                                  "'policies/hashicorp/CIS-Policy-Set-for-AWS-Terraform/1.0.1')"}},
                                      'required': ['terraform_policy_id']}}}


@function_tool(schema=_SCHEMA_get_policy_details)
def get_policy_details(**kwargs: object) -> str:
    """Fetches up-to-date documentation for a specific policy from the Terraform registry. You must call 'search_policies' first to obtain the exact terraform_policy_id required to use this tool."""
    return call_tool('get_policy_details', kwargs)


_SCHEMA_get_provider_capabilities = {   'type': 'function',
    'function': {   'name': 'get_provider_capabilities',
                    'description': 'Get the capabilities of a Terraform provider '
                                   'including the types of resources, data sources, '
                                   'functions, guides, and other features it '
                                   'supports.\n'
                                   'This tool analyzes the provider documentation to '
                                   'determine what types of capabilities are '
                                   'available:\n'
                                   '- resources: Infrastructure resources that can be '
                                   'created/managed\n'
                                   '- data-sources: Read-only data sources for '
                                   'querying existing infrastructure  \n'
                                   '- functions: Provider-specific functions for data '
                                   'transformation\n'
                                   '- guides: Documentation guides and tutorials for '
                                   'using the provider\n'
                                   '- actions: Available provider actions (if any)\n'
                                   '- ephemeral resources: Temporary resources for '
                                   'credentials and tokens\n'
                                   '- list-resources: List resources for querying '
                                   'existing cloud resources (Terraform Search)\n'
                                   '\n'
                                   'Returns a summary with counts and examples for '
                                   'each capability type.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'name': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'name '
                                                                                   'of '
                                                                                   'the '
                                                                                   'Terraform '
                                                                                   'provider, '
                                                                                   'e.g., '
                                                                                   "'aws', "
                                                                                   "'azurerm', "
                                                                                   "'google', "
                                                                                   'etc.'},
                                                        'namespace': {   'type': 'string',
                                                                         'description': 'The '
                                                                                        'namespace '
                                                                                        'of '
                                                                                        'the '
                                                                                        'Terraform '
                                                                                        'provider, '
                                                                                        'typically '
                                                                                        'the '
                                                                                        'name '
                                                                                        'of '
                                                                                        'the '
                                                                                        'company, '
                                                                                        'or '
                                                                                        'their '
                                                                                        'GitHub '
                                                                                        'organization '
                                                                                        'name '
                                                                                        'that '
                                                                                        'created '
                                                                                        'the '
                                                                                        'provider '
                                                                                        'e.g., '
                                                                                        "'hashicorp'"},
                                                        'version': {   'type': 'string',
                                                                       'description': 'The '
                                                                                      'version '
                                                                                      'of '
                                                                                      'the '
                                                                                      'provider '
                                                                                      'to '
                                                                                      'analyze '
                                                                                      '(defaults '
                                                                                      'to '
                                                                                      "'latest')"}},
                                      'required': ['namespace', 'name']}}}


@function_tool(schema=_SCHEMA_get_provider_capabilities)
def get_provider_capabilities(**kwargs: object) -> str:
    """Get the capabilities of a Terraform provider including the types of resources, data sources, functions, guides, and other features it supports."""
    return call_tool('get_provider_capabilities', kwargs)


_SCHEMA_get_provider_details = {   'type': 'function',
    'function': {   'name': 'get_provider_details',
                    'description': 'Fetches up-to-date documentation for a specific '
                                   'service from a Terraform provider. \n'
                                   "You must call 'search_providers' tool first to "
                                   'obtain the exact tfprovider-compatible '
                                   'provider_doc_id required to use this tool.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'provider_doc_id': {   'type': 'string',
                                                                               'description': 'Exact '
                                                                                              'tfprovider-compatible '
                                                                                              'provider_doc_id, '
                                                                                              '(e.g., '
                                                                                              "'8894603', "
                                                                                              "'8906901') "
                                                                                              'retrieved '
                                                                                              'from '
                                                                                              "'search_providers'"}},
                                      'required': ['provider_doc_id']}}}


@function_tool(schema=_SCHEMA_get_provider_details)
def get_provider_details(**kwargs: object) -> str:
    """Fetches up-to-date documentation for a specific service from a Terraform provider."""
    return call_tool('get_provider_details', kwargs)


_SCHEMA_get_timestamp = {   'type': 'function',
    'function': {   'name': 'get_timestamp',
                    'description': 'Get the timestamp for the time.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'time': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'time '
                                                                                   'to '
                                                                                   'get '
                                                                                   'the '
                                                                                   'timestamp. '
                                                                                   'Format: '
                                                                                   'YYYY-MM-DD '
                                                                                   'HH:mm:ss.SSS'}}}}}


@function_tool(schema=_SCHEMA_get_timestamp)
def get_timestamp(**kwargs: object) -> str:
    """Get the timestamp for the time."""
    return call_tool('get_timestamp', kwargs)


_SCHEMA_get_week_year = {   'type': 'function',
    'function': {   'name': 'get_week_year',
                    'description': 'Get the week and isoWeek of the year.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'date': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'date '
                                                                                   'to '
                                                                                   'get '
                                                                                   'the '
                                                                                   'week '
                                                                                   'and '
                                                                                   'isoWeek '
                                                                                   'of '
                                                                                   'the '
                                                                                   'year. '
                                                                                   'e.g. '
                                                                                   '2025-03-23'}}}}}


@function_tool(schema=_SCHEMA_get_week_year)
def get_week_year(**kwargs: object) -> str:
    """Get the week and isoWeek of the year."""
    return call_tool('get_week_year', kwargs)


_SCHEMA_is_prime = {   'type': 'function',
    'function': {   'name': 'is_prime',
                    'description': 'Check if a number is prime',
                    'parameters': {   'properties': {'n': {'type': 'integer'}},
                                      'required': ['n'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_is_prime)
def is_prime(**kwargs: object) -> str:
    """Check if a number is prime"""
    return call_tool('is_prime', kwargs)


_SCHEMA_lcm = {   'type': 'function',
    'function': {   'name': 'lcm',
                    'description': 'Calculate least common multiple of two integers',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_lcm)
def lcm(**kwargs: object) -> str:
    """Calculate least common multiple of two integers"""
    return call_tool('lcm', kwargs)


_SCHEMA_list_bible_books = {   'type': 'function',
    'function': {   'name': 'list_bible_books',
                    'description': '\n'
                                   '    Get list of books for a specific Bible '
                                   'translation.\n'
                                   '\n'
                                   '    Args:\n'
                                   '        translation: Translation identifier '
                                   '(default: "web")\n'
                                   '\n'
                                   '    Returns:\n'
                                   '        List of books with their identifiers and '
                                   'names\n'
                                   '    ',
                    'parameters': {   'properties': {   'translation': {   'default': 'web',
                                                                           'type': 'string'}},
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_list_bible_books)
def list_bible_books(**kwargs: object) -> str:
    """Get list of books for a specific Bible translation."""
    return call_tool('list_bible_books', kwargs)


_SCHEMA_list_bible_chapters = {   'type': 'function',
    'function': {   'name': 'list_bible_chapters',
                    'description': '\n'
                                   '    Get chapters for a specific Bible book.\n'
                                   '\n'
                                   '    Args:\n'
                                   '        book: Book identifier (e.g., "JHN" for '
                                   'John, "GEN" for Genesis)\n'
                                   '        translation: Translation identifier '
                                   '(default: "web")\n'
                                   '\n'
                                   '    Returns:\n'
                                   '        List of chapters for the specified book\n'
                                   '    ',
                    'parameters': {   'properties': {   'book': {'type': 'string'},
                                                        'translation': {   'default': 'web',
                                                                           'type': 'string'}},
                                      'required': ['book'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_list_bible_chapters)
def list_bible_chapters(**kwargs: object) -> str:
    """Get chapters for a specific Bible book."""
    return call_tool('list_bible_chapters', kwargs)


_SCHEMA_list_bible_translations = {   'type': 'function',
    'function': {   'name': 'list_bible_translations',
                    'description': '\n'
                                   '    Get list of all available Bible translations.\n'
                                   '\n'
                                   '    Returns:\n'
                                   '        List of available translations with their '
                                   'identifiers and names\n'
                                   '    ',
                    'parameters': {'properties': {}, 'type': 'object'}}}


@function_tool(schema=_SCHEMA_list_bible_translations)
def list_bible_translations(**kwargs: object) -> str:
    """Get list of all available Bible translations."""
    return call_tool('list_bible_translations', kwargs)


_SCHEMA_log = {   'type': 'function',
    'function': {   'name': 'log',
                    'description': 'Calculate logarithm of a number with optional base '
                                   '(default: natural log)',
                    'parameters': {   'properties': {   'x': {'type': 'number'},
                                                        'base': {   'default': 2.718281828459045,
                                                                    'type': 'number'}},
                                      'required': ['x'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_log)
def log(**kwargs: object) -> str:
    """Calculate logarithm of a number with optional base (default: natural log)"""
    return call_tool('log', kwargs)


_SCHEMA_mul = {   'type': 'function',
    'function': {   'name': 'mul',
                    'description': 'Multiply two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_mul)
def mul(**kwargs: object) -> str:
    """Multiply two numbers"""
    return call_tool('mul', kwargs)


_SCHEMA_mul_float = {   'type': 'function',
    'function': {   'name': 'mul_float',
                    'description': 'Multiply two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'number'},
                                                        'b': {'type': 'number'}},
                                      'type': 'object',
                                      'required': ['a', 'b']}}}


@function_tool(schema=_SCHEMA_mul_float)
def mul_float(**kwargs: object) -> str:
    """Multiply two numbers"""
    return call_tool('mul_float', kwargs)


_SCHEMA_power = {   'type': 'function',
    'function': {   'name': 'power',
                    'description': 'Raise a number to a power',
                    'parameters': {   'properties': {   'base': {'type': 'number'},
                                                        'exponent': {'type': 'number'}},
                                      'required': ['base', 'exponent'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_power)
def power(**kwargs: object) -> str:
    """Raise a number to a power"""
    return call_tool('power', kwargs)


_SCHEMA_quadratic_roots = {   'type': 'function',
    'function': {   'name': 'quadratic_roots',
                    'description': '\n'
                                   '    Solve quadratic equation ax² + bx + c = 0\n'
                                   '    Returns a tuple of roots (real or complex)\n'
                                   '    ',
                    'parameters': {   'properties': {   'a': {'type': 'number'},
                                                        'b': {'type': 'number'},
                                                        'c': {'type': 'number'}},
                                      'required': ['a', 'b', 'c'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_quadratic_roots)
def quadratic_roots(**kwargs: object) -> str:
    """Solve quadratic equation ax² + bx + c = 0"""
    return call_tool('quadratic_roots', kwargs)


_SCHEMA_radians_to_degrees = {   'type': 'function',
    'function': {   'name': 'radians_to_degrees',
                    'description': 'Convert radians to degrees',
                    'parameters': {   'properties': {'radians': {'type': 'number'}},
                                      'required': ['radians'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_radians_to_degrees)
def radians_to_degrees(**kwargs: object) -> str:
    """Convert radians to degrees"""
    return call_tool('radians_to_degrees', kwargs)


_SCHEMA_relative_time = {   'type': 'function',
    'function': {   'name': 'relative_time',
                    'description': 'Get the relative time from now.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'time': {   'type': 'string',
                                                                    'description': 'The '
                                                                                   'time '
                                                                                   'to '
                                                                                   'get '
                                                                                   'the '
                                                                                   'relative '
                                                                                   'time '
                                                                                   'from '
                                                                                   'now. '
                                                                                   'Format: '
                                                                                   'YYYY-MM-DD '
                                                                                   'HH:mm:ss'}},
                                      'required': ['time']}}}


@function_tool(schema=_SCHEMA_relative_time)
def relative_time(**kwargs: object) -> str:
    """Get the relative time from now."""
    return call_tool('relative_time', kwargs)


_SCHEMA_reverse_text_tool = {   'type': 'function',
    'function': {   'name': 'reverse_text_tool',
                    'description': 'Reverse text tool',
                    'parameters': {   'type': 'object',
                                      'properties': {'text': {'type': 'string'}},
                                      'required': ['text'],
                                      'additionalProperties': False}}}


@function_tool(schema=_SCHEMA_reverse_text_tool)
def reverse_text_tool(**kwargs: object) -> str:
    """Reverse text tool"""
    return call_tool('reverse_text_tool', kwargs)


_SCHEMA_scrape_recent_category_papers = {   'type': 'function',
    'function': {   'name': 'scrape_recent_category_papers',
                    'description': "[크롤링] 특정 카테고리의 'recent' 페이지를 스크랩하여 최신 논문 목록을 "
                                   '가져옵니다.',
                    'parameters': {   'properties': {   'category': {'type': 'string'},
                                                        'max_results': {   'default': 10,
                                                                           'type': 'integer'}},
                                      'required': ['category'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_scrape_recent_category_papers)
def scrape_recent_category_papers(**kwargs: object) -> str:
    """[크롤링] 특정 카테고리의 'recent' 페이지를 스크랩하여 최신 논문 목록을 가져옵니다."""
    return call_tool('scrape_recent_category_papers', kwargs)


_SCHEMA_search_books_tool = {   'type': 'function',
    'function': {   'name': 'search_books_tool',
                    'description': '\n'
                                   '    Searches for books with the query provided by '
                                   'the user.\n'
                                   '    ',
                    'parameters': {   'properties': {'query': {'type': 'string'}},
                                      'required': ['query'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_search_books_tool)
def search_books_tool(**kwargs: object) -> str:
    """Searches for books with the query provided by the user."""
    return call_tool('search_books_tool', kwargs)


_SCHEMA_search_modules = {   'type': 'function',
    'function': {   'name': 'search_modules',
                    'description': 'Resolves a Terraform module name to obtain a '
                                   'compatible module_id for the get_module_details '
                                   'tool and returns a list of matching Terraform '
                                   'modules.\n'
                                   'You MUST call this function before '
                                   "'get_module_details' to obtain a valid and "
                                   'compatible module_id.\n'
                                   'When selecting the best match, consider the '
                                   'following:\n'
                                   '\t- Name similarity to the query\n'
                                   '\t- Description relevance\n'
                                   '\t- Verification status (verified)\n'
                                   '\t- Download counts (popularity)\n'
                                   'Return the selected module_id and explain your '
                                   'choice. If there are multiple good matches, '
                                   'mention this but proceed with the most relevant '
                                   'one.\n'
                                   'If no modules were found, reattempt the search '
                                   'with a new moduleName query.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'current_offset': {   'type': 'integer',
                                                                              'default': 0,
                                                                              'minimum': 0,
                                                                              'description': 'Current '
                                                                                             'offset '
                                                                                             'for '
                                                                                             'pagination'},
                                                        'module_query': {   'type': 'string',
                                                                            'description': 'The '
                                                                                           'query '
                                                                                           'to '
                                                                                           'search '
                                                                                           'for '
                                                                                           'Terraform '
                                                                                           'modules.'}},
                                      'required': ['module_query']}}}


@function_tool(schema=_SCHEMA_search_modules)
def search_modules(**kwargs: object) -> str:
    """Resolves a Terraform module name to obtain a compatible module_id for the get_module_details tool and returns a list of matching Terraform modules."""
    return call_tool('search_modules', kwargs)


_SCHEMA_search_policies = {   'type': 'function',
    'function': {   'name': 'search_policies',
                    'description': 'Searches for Terraform policies based on a query '
                                   'string.\n'
                                   'This tool returns a list of matching policies, '
                                   'which can be used to retrieve detailed policy '
                                   "information using the 'get_policy_details' tool.\n"
                                   'You MUST call this function before '
                                   "'get_policy_details' to obtain a valid "
                                   'terraform_policy_id.\n'
                                   'When selecting the best match, consider the '
                                   'following:\n'
                                   '\t- Name similarity to the query\n'
                                   '\t- Title relevance\n'
                                   '\t- Verification status (verified)\n'
                                   '\t- Download counts (popularity)\n'
                                   'Return the selected policyID and explain your '
                                   'choice. If there are multiple good matches, '
                                   'mention this but proceed with the most relevant '
                                   'one.\n'
                                   'If no policies were found, reattempt the search '
                                   'with a new policy_query.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'policy_query': {   'type': 'string',
                                                                            'description': 'The '
                                                                                           'query '
                                                                                           'to '
                                                                                           'search '
                                                                                           'for '
                                                                                           'Terraform '
                                                                                           'modules.'}},
                                      'required': ['policy_query']}}}


@function_tool(schema=_SCHEMA_search_policies)
def search_policies(**kwargs: object) -> str:
    """Searches for Terraform policies based on a query string."""
    return call_tool('search_policies', kwargs)


_SCHEMA_search_providers = {   'type': 'function',
    'function': {   'name': 'search_providers',
                    'description': 'This tool retrieves a list of potential documents '
                                   "based on the 'service_slug' and "
                                   "'provider_document_type' provided.\n"
                                   'You MUST call this function before '
                                   "'get_provider_details' to obtain a valid "
                                   "tfprovider-compatible 'provider_doc_id'.\n"
                                   'Use the most relevant single word as the search '
                                   "query for 'service_slug', if unsure about the "
                                   "'service_slug', use the 'provider_name' for its "
                                   'value.\n'
                                   'When selecting the best match, consider the '
                                   'following:\n'
                                   '\t- Title similarity to the query\n'
                                   '\t- Category relevance\n'
                                   "Return the selected 'provider_doc_id' and explain "
                                   'your choice.\n'
                                   'If there are multiple good matches, mention this '
                                   'but proceed with the most relevant one.',
                    'parameters': {   'type': 'object',
                                      'properties': {   'provider_document_type': {   'type': 'string',
                                                                                      'description': 'The '
                                                                                                     'type '
                                                                                                     'of '
                                                                                                     'the '
                                                                                                     'document '
                                                                                                     'to '
                                                                                                     'retrieve,\n'
                                                                                                     'for '
                                                                                                     'general '
                                                                                                     'overview '
                                                                                                     'of '
                                                                                                     'the '
                                                                                                     'provider '
                                                                                                     'use '
                                                                                                     "'overview',\n"
                                                                                                     'for '
                                                                                                     'guidance '
                                                                                                     'on '
                                                                                                     'upgrading '
                                                                                                     'a '
                                                                                                     'provider '
                                                                                                     'or '
                                                                                                     'custom '
                                                                                                     'configuration '
                                                                                                     'information '
                                                                                                     'use '
                                                                                                     "'guides',\n"
                                                                                                     'for '
                                                                                                     'deploying '
                                                                                                     'resources '
                                                                                                     'use '
                                                                                                     "'resources', "
                                                                                                     'for '
                                                                                                     'reading '
                                                                                                     'pre-deployed '
                                                                                                     'resources '
                                                                                                     'use '
                                                                                                     "'data-sources',\n"
                                                                                                     'for '
                                                                                                     'functions '
                                                                                                     'use '
                                                                                                     "'functions',\n"
                                                                                                     'for '
                                                                                                     'Terraform '
                                                                                                     'actions '
                                                                                                     'use '
                                                                                                     "'actions',\n"
                                                                                                     'for '
                                                                                                     'listing '
                                                                                                     'resources '
                                                                                                     'using '
                                                                                                     'Terraform '
                                                                                                     'Search '
                                                                                                     'use '
                                                                                                     "'list-resources'",
                                                                                      'enum': [   'resources',
                                                                                                  'data-sources',
                                                                                                  'functions',
                                                                                                  'guides',
                                                                                                  'overview',
                                                                                                  'actions',
                                                                                                  'list-resources']},
                                                        'provider_name': {   'type': 'string',
                                                                             'description': 'The '
                                                                                            'name '
                                                                                            'of '
                                                                                            'the '
                                                                                            'Terraform '
                                                                                            'provider '
                                                                                            'to '
                                                                                            'perform '
                                                                                            'the '
                                                                                            'read '
                                                                                            'or '
                                                                                            'deployment '
                                                                                            'operation'},
                                                        'provider_namespace': {   'type': 'string',
                                                                                  'description': 'The '
                                                                                                 'publisher '
                                                                                                 'of '
                                                                                                 'the '
                                                                                                 'Terraform '
                                                                                                 'provider, '
                                                                                                 'typically '
                                                                                                 'the '
                                                                                                 'name '
                                                                                                 'of '
                                                                                                 'the '
                                                                                                 'company, '
                                                                                                 'or '
                                                                                                 'their '
                                                                                                 'GitHub '
                                                                                                 'organization '
                                                                                                 'name '
                                                                                                 'that '
                                                                                                 'created '
                                                                                                 'the '
                                                                                                 'provider'},
                                                        'provider_version': {   'type': 'string',
                                                                                'description': 'The '
                                                                                               'version '
                                                                                               'of '
                                                                                               'the '
                                                                                               'Terraform '
                                                                                               'provider '
                                                                                               'to '
                                                                                               'retrieve '
                                                                                               'in '
                                                                                               'the '
                                                                                               'format '
                                                                                               "'x.y.z', "
                                                                                               'or '
                                                                                               "'latest' "
                                                                                               'to '
                                                                                               'get '
                                                                                               'the '
                                                                                               'latest '
                                                                                               'version'},
                                                        'service_slug': {   'type': 'string',
                                                                            'description': 'The '
                                                                                           'slug '
                                                                                           'of '
                                                                                           'the '
                                                                                           'service '
                                                                                           'you '
                                                                                           'want '
                                                                                           'to '
                                                                                           'deploy '
                                                                                           'or '
                                                                                           'read '
                                                                                           'using '
                                                                                           'the '
                                                                                           'Terraform '
                                                                                           'provider, '
                                                                                           'prefer '
                                                                                           'using '
                                                                                           'a '
                                                                                           'single '
                                                                                           'word, '
                                                                                           'use '
                                                                                           'underscores '
                                                                                           'for '
                                                                                           'multiple '
                                                                                           'words '
                                                                                           'and '
                                                                                           'if '
                                                                                           'unsure '
                                                                                           'about '
                                                                                           'the '
                                                                                           'service_slug, '
                                                                                           'use '
                                                                                           'the '
                                                                                           'provider_name '
                                                                                           'for '
                                                                                           'its '
                                                                                           'value'}},
                                      'required': [   'provider_name',
                                                      'provider_namespace',
                                                      'service_slug',
                                                      'provider_document_type']}}}


@function_tool(schema=_SCHEMA_search_providers)
def search_providers(**kwargs: object) -> str:
    """This tool retrieves a list of potential documents based on the 'service_slug' and 'provider_document_type' provided."""
    return call_tool('search_providers', kwargs)


_SCHEMA_sin = {   'type': 'function',
    'function': {   'name': 'sin',
                    'description': 'Calculate sine of an angle in radians',
                    'parameters': {   'properties': {'x': {'type': 'number'}},
                                      'required': ['x'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_sin)
def sin(**kwargs: object) -> str:
    """Calculate sine of an angle in radians"""
    return call_tool('sin', kwargs)


_SCHEMA_square_root = {   'type': 'function',
    'function': {   'name': 'square_root',
                    'description': 'Calculate square root of a number',
                    'parameters': {   'properties': {'x': {'type': 'number'}},
                                      'required': ['x'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_square_root)
def square_root(**kwargs: object) -> str:
    """Calculate square root of a number"""
    return call_tool('square_root', kwargs)


_SCHEMA_sub = {   'type': 'function',
    'function': {   'name': 'sub',
                    'description': 'Subtract two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'integer'},
                                                        'b': {'type': 'integer'}},
                                      'required': ['a', 'b'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_sub)
def sub(**kwargs: object) -> str:
    """Subtract two numbers"""
    return call_tool('sub', kwargs)


_SCHEMA_sub_float = {   'type': 'function',
    'function': {   'name': 'sub_float',
                    'description': 'Subtract two numbers',
                    'parameters': {   'properties': {   'a': {'type': 'number'},
                                                        'b': {'type': 'number'}},
                                      'type': 'object',
                                      'required': ['a', 'b']}}}


@function_tool(schema=_SCHEMA_sub_float)
def sub_float(**kwargs: object) -> str:
    """Subtract two numbers"""
    return call_tool('sub_float', kwargs)


_SCHEMA_tan = {   'type': 'function',
    'function': {   'name': 'tan',
                    'description': 'Calculate tangent of an angle in radians',
                    'parameters': {   'properties': {'x': {'type': 'number'}},
                                      'required': ['x'],
                                      'type': 'object'}}}


@function_tool(schema=_SCHEMA_tan)
def tan(**kwargs: object) -> str:
    """Calculate tangent of an angle in radians"""
    return call_tool('tan', kwargs)
