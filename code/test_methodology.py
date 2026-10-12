import unittest
from tool_protocol import parse_calls
from token_budget import select_fresh
from reflection_targets import trajectory
from common import reflection_feedback

class Methodology(unittest.TestCase):
    def test_native_multiline_without_json_escaping(self):
        raw='think</think><tool_call>\n<function=write_file>\n<parameter=path>\n/app/a.py\n</parameter>\n<parameter=content>\nprint("quotes")\nprint(2)\n</parameter>\n</function>\n</tool_call>'
        calls=parse_calls(raw);self.assertEqual(calls[0]['arguments']['content'],'print("quotes")\nprint(2)')
    def test_duplicate_argument_rejected(self):
        with self.assertRaises(ValueError):parse_calls('<tool_call><function=bash><parameter=command>x</parameter><parameter=command>y</parameter></function></tool_call>')
    def test_exact_budget_preserves_whole_targets(self):
        rows=[{'id':str(i),'token_ids':list(range(7))} for i in range(10)]
        selected,receipt=select_fresh(rows,23,1)
        self.assertEqual([len(r.get('supervised_positions',r['token_ids'])) for r in selected],[7,7,7,2])
        self.assertEqual(receipt['boundary_target'],'3')
    def test_no_false_execution_success(self):
        text=reflection_feedback({'execution':{'status':'verifier_rejected_submission','saved':False,
            'file_exists':True,'blend_header_valid':False},'finish_reason':'stop','output_tokens':1,
            'verifier_failure':{'reason':'corrupt or unreadable submission'},'validation_status':'invalid_submission'})
        self.assertIn('corrupt or unreadable submission',text);self.assertIn('invalid_submission',text)
if __name__=='__main__':unittest.main()
