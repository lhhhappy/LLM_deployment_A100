import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/pod/verify/kernel_combo_cost.py'
spec=importlib.util.spec_from_file_location('combo_cost',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CostContract(unittest.TestCase):
    def test_flush_requires_positive_json_ack(self):
        for body in (b'{}', b'', b'Cache flushed.', b'{"success":false}', b'{"success":1}'):
            with self.subTest(body=body), patch.object(m, 'post', return_value=body), self.assertRaises(ValueError):
                m.flush()
        with patch.object(m, 'post', return_value=b'{"success":true}'):
            self.assertEqual(m.flush(), '{"success":true}')
    def valid(self):
        return {'meta_info':{'prompt_tokens':256,'completion_tokens':2,'cached_tokens':0,
                            'output_token_logprobs':[[-1.,12,None],[-2.,13,None]]}}
    def test_rejects_invalid_cost_samples(self):
        self.assertEqual(m.validate(self.valid(),256,2)['cached_tokens'],0)
        for field,value in [('prompt_tokens',255),('completion_tokens',1),('cached_tokens',64),
                            ('output_token_logprobs',[[-1.,12,None]]),
                            ('output_token_logprobs',[[float('nan'),12,None],[-2.,13,None]])]:
            response=self.valid();response['meta_info'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate(response,256,2)
    def test_repeatable_independent_requests(self):
        a=m.inputs(256,32)
        self.assertEqual(a,m.inputs(256,32))
        self.assertEqual(len({tuple(x) for x in a}),32)
        self.assertEqual({len(x) for x in a},{256})
        self.assertTrue(all(len(set(x))>1 for x in a))

if __name__=='__main__':unittest.main()
