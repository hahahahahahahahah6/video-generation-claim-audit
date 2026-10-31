import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from video_generation_claim_audit.audit import claims, evidence, audit, remove_matches
from video_generation_claim_audit.phrases import VIDEOGENERATION, NOVIDEOGENERATION

FIXTURES = Path(__file__).parent / 'fixtures'

class Claims(unittest.TestCase):
    def test_fences(self):
        self.assertEqual(claims('   ```text\ngenerates videos\n   ```\n~~~\nvideo generation\n~~~'), (False, False))
    def test_inline(self):
        self.assertEqual(claims('`generates videos`'), (True, False))
    def test_negative_removal(self):
        self.assertEqual(claims('does not support video generation'), (False, True))
    def test_hyphens(self):
        self.assertEqual(claims('text-to-video'), (True, False))
    def test_contraction(self):
        self.assertEqual(claims("doesn't support video generation"), (False, True))
    def test_paragraph_join(self):
        self.assertEqual(claims('supports text to\nvideo generation'), (True, False))
        self.assertEqual(claims('generates\n\nvideos'), (False, False))
    def test_longest_first(self):
        remaining, found = remove_matches('generates videos from text prompts', ('generates videos', 'generates videos from text prompts'))
        self.assertTrue(found)
        self.assertEqual(remaining.strip(), '')
    def test_boundaries(self):
        self.assertEqual(claims('regenerates videoscope'), (False, False))
    def test_order(self):
        for values in (VIDEOGENERATION, NOVIDEOGENERATION):
            self.assertEqual(list(values), sorted(set(values), key=lambda s: (-len(s), s)))
    def test_all_templates(self):
        for phrase in VIDEOGENERATION:
            self.assertEqual(claims(phrase), (True, False), phrase)
        for phrase in NOVIDEOGENERATION:
            self.assertEqual(claims(phrase), (False, True), phrase)

class Evidence(unittest.TestCase):
    pass

class Integration(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'video_generation_claim_audit', *map(str,args)], text=True, capture_output=True)
    def test_fixtures(self):
        for name, severity, code in [('good',None,0),('bad','HIGH',2),('testonly','MEDIUM',0),('no-video-generation','HIGH',2),('no-claim',None,0)]:
            with self.subTest(name=name):
                result=self.cli(FIXTURES/name,'--json')
                self.assertEqual(result.returncode,code,result.stderr)
                data=json.loads(result.stdout)
                self.assertEqual(set(data),{'verdict','findings','files_scanned'})
                self.assertEqual(data['files_scanned'],2)
                if severity:
                    rule='documented-no-video-generation-claim-false' if name=='no-video-generation' else 'documented-video-generation-claim-false'
                    self.assertEqual(len(data['findings']),1)
                    self.assertEqual(data['findings'][0]['severity'],severity)
                    self.assertEqual(data['findings'][0]['rule'],rule)
                else:
                    self.assertEqual(data['findings'],[])
    def test_medium(self):
        self.assertEqual(self.cli(FIXTURES/'testonly','--fail-on','MEDIUM').returncode,2)
    def test_ignore(self):
        self.assertEqual(audit(FIXTURES/'bad', ['README.*'])['findings'],[])
    def test_verbose(self):
        self.assertIn('Files scanned: 2',self.cli(FIXTURES/'good','--verbose').stdout)
    def test_version(self):
        result=self.cli('--version')
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'0.1.0'))
    def test_invalid_root(self):
        self.assertEqual(self.cli(FIXTURES/'absent').returncode,1)
        self.assertEqual(self.cli(FIXTURES/'good'/'app.py').returncode,1)
    def test_usage(self):
        self.assertEqual(self.cli('--unknown').returncode,1)
    def test_self(self):
        self.assertEqual(audit(Path(__file__).parent.parent)['findings'],[])
    def test_exclusions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'README.md').write_text('generates videos')
            for name in ['docs','examples','fixtures','tests']:
                (root/name).mkdir()
                (root/name/'impl.py').write_text('from diffusers import AnimateDiffPipeline\nAnimateDiffPipeline()')
            (root/'link.py').symlink_to(root/'tests'/'impl.py')
            (root/'linked').symlink_to(root/'tests',target_is_directory=True)
            self.assertEqual(audit(root)['findings'][0]['severity'],'MEDIUM')
            (root/'tests'/'impl.py').unlink()
            self.assertEqual(audit(root)['findings'][0]['severity'],'HIGH')
    def test_nested_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'sub'/'docs').mkdir(parents=True)
            (root/'sub'/'docs'/'guide.md').write_text('generates videos')
            self.assertEqual(len(audit(root)['findings']),1)

def check_direct(self):
    self.assertEqual(evidence('import diffusers\ndiffusers.AnimateDiffPipeline()'), True)
Evidence.test_direct = check_direct

def check_aliased(self):
    self.assertEqual(evidence('import diffusers as d\nd.AnimateDiffPipeline()'), True)
Evidence.test_aliased = check_aliased

def check_from_import(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline as A\nA()'), True)
Evidence.test_from_import = check_from_import

def check_loader(self):
    self.assertEqual(evidence('import diffusers\ndiffusers.CogVideoXPipeline.from_pretrained("x")'), True)
Evidence.test_loader = check_loader

def check_loader_alias(self):
    self.assertEqual(evidence('from diffusers import CogVideoXPipeline as C\nC.from_pretrained("x")'), True)
Evidence.test_loader_alias = check_loader_alias

def check_single_file(self):
    self.assertEqual(evidence('from diffusers import CogVideoXPipeline as C\nC.from_single_file("x")'), True)
Evidence.test_single_file = check_single_file

def check_positional(self):
    self.assertEqual(evidence('from transformers import pipeline\npipeline("text-to-video")'), True)
Evidence.test_positional = check_positional

def check_keyword(self):
    self.assertEqual(evidence('import transformers as t\nt.pipeline(task="video-generation")'), True)
Evidence.test_keyword = check_keyword

def check_hosted_videos(self):
    self.assertEqual(evidence('client.videos.generate(prompt="x")'), True)
Evidence.test_hosted_videos = check_hosted_videos

def check_hosted_video(self):
    self.assertEqual(evidence('client.video.generate(prompt="x")'), True)
Evidence.test_hosted_video = check_hosted_video

def check_TextToVideoSDPipeline(self):
    self.assertEqual(evidence('from diffusers import TextToVideoSDPipeline\nTextToVideoSDPipeline()'), True)
Evidence.test_TextToVideoSDPipeline = check_TextToVideoSDPipeline

def check_StableVideoDiffusionPipeline(self):
    self.assertEqual(evidence('from diffusers import StableVideoDiffusionPipeline\nStableVideoDiffusionPipeline()'), True)
Evidence.test_StableVideoDiffusionPipeline = check_StableVideoDiffusionPipeline

def check_I2VGenXLPipeline(self):
    self.assertEqual(evidence('from diffusers import I2VGenXLPipeline\nI2VGenXLPipeline()'), True)
Evidence.test_I2VGenXLPipeline = check_I2VGenXLPipeline

def check_AnimateDiffControlNetPipeline(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffControlNetPipeline\nAnimateDiffControlNetPipeline()'), True)
Evidence.test_AnimateDiffControlNetPipeline = check_AnimateDiffControlNetPipeline

def check_TextToVideoZeroPipeline(self):
    self.assertEqual(evidence('from diffusers import TextToVideoZeroPipeline\nTextToVideoZeroPipeline()'), True)
Evidence.test_TextToVideoZeroPipeline = check_TextToVideoZeroPipeline

def check_CogVideoXImageToVideoPipeline(self):
    self.assertEqual(evidence('from diffusers import CogVideoXImageToVideoPipeline\nCogVideoXImageToVideoPipeline()'), True)
Evidence.test_CogVideoXImageToVideoPipeline = check_CogVideoXImageToVideoPipeline

def check_CogVideoXVideoToVideoPipeline(self):
    self.assertEqual(evidence('from diffusers import CogVideoXVideoToVideoPipeline\nCogVideoXVideoToVideoPipeline()'), True)
Evidence.test_CogVideoXVideoToVideoPipeline = check_CogVideoXVideoToVideoPipeline

def check_VideoCrafterPipeline(self):
    self.assertEqual(evidence('from diffusers import VideoCrafterPipeline\nVideoCrafterPipeline()'), True)
Evidence.test_VideoCrafterPipeline = check_VideoCrafterPipeline

def check_bare(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline'), False)
Evidence.test_bare = check_bare

def check_pil(self):
    self.assertEqual(evidence('from PIL import ImageDraw\nImageDraw.Draw(image).rectangle((0,0,1,1))'), False)
Evidence.test_pil = check_pil

def check_plot(self):
    self.assertEqual(evidence('import matplotlib.pyplot as plt\nplt.plot([1,2])'), False)
Evidence.test_plot = check_plot

def check_writer(self):
    self.assertEqual(evidence('import cv2\ncv2.VideoWriter("x",0,30,(2,2))'), False)
Evidence.test_writer = check_writer

def check_image(self):
    self.assertEqual(evidence('from diffusers import StableDiffusionPipeline\nStableDiffusionPipeline.from_pretrained("x")'), False)
Evidence.test_image = check_image

def check_generic_diffusion(self):
    self.assertEqual(evidence('from diffusers import DiffusionPipeline\nDiffusionPipeline.from_pretrained("x")'), False)
Evidence.test_generic_diffusion = check_generic_diffusion

def check_speech(self):
    self.assertEqual(evidence('from transformers import pipeline\npipeline("text-to-speech")'), False)
Evidence.test_speech = check_speech

def check_detection(self):
    self.assertEqual(evidence('from ultralytics import YOLO\nYOLO("x")()'), False)
Evidence.test_detection = check_detection

def check_segmentation(self):
    self.assertEqual(evidence('model.segment(image)'), False)
Evidence.test_segmentation = check_segmentation

def check_classification(self):
    self.assertEqual(evidence('model.classify(image)'), False)
Evidence.test_classification = check_classification

def check_face(self):
    self.assertEqual(evidence('cv2.CascadeClassifier("x").detectMultiScale(image)'), False)
Evidence.test_face = check_face

def check_hand(self):
    self.assertEqual(evidence('mp.solutions.hands.Hands()'), False)
Evidence.test_hand = check_hand

def check_pose(self):
    self.assertEqual(evidence('mp.solutions.pose.Pose()'), False)
Evidence.test_pose = check_pose

def check_tracking(self):
    self.assertEqual(evidence('cv2.TrackerCSRT_create()'), False)
Evidence.test_tracking = check_tracking

def check_depth(self):
    self.assertEqual(evidence('model.estimate_depth(image)'), False)
Evidence.test_depth = check_depth

def check_flow(self):
    self.assertEqual(evidence('cv2.calcOpticalFlowFarneback(a,b,None,0.5,3,15,3,5,1.2,0)'), False)
Evidence.test_flow = check_flow

def check_super_resolution(self):
    self.assertEqual(evidence('cv2.dnn_superres.DnnSuperResImpl_create()'), False)
Evidence.test_super_resolution = check_super_resolution

def check_ocr(self):
    self.assertEqual(evidence('reader.readtext(image)'), False)
Evidence.test_ocr = check_ocr

def check_barcode(self):
    self.assertEqual(evidence('pyzbar.decode(image)'), False)
Evidence.test_barcode = check_barcode

def check_shadow(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline\ndef AnimateDiffPipeline(): pass\nAnimateDiffPipeline()'), False)
Evidence.test_shadow = check_shadow

def check_class_shadow(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline\nclass AnimateDiffPipeline: pass\nAnimateDiffPipeline()'), False)
Evidence.test_class_shadow = check_class_shadow

def check_assignment_shadow(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline\nAnimateDiffPipeline = factory\nAnimateDiffPipeline()'), False)
Evidence.test_assignment_shadow = check_assignment_shadow

def check_string(self):
    self.assertEqual(evidence('x = "video generation AnimateDiffPipeline"'), False)
Evidence.test_string = check_string

def check_subprocess(self):
    self.assertEqual(evidence('import subprocess\nsubprocess.run(["ffmpeg", "x"])'), False)
Evidence.test_subprocess = check_subprocess

def check_invalid(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline\nAnimateDiffPipeline(('), False)
Evidence.test_invalid = check_invalid

def check_unbound(self):
    self.assertEqual(evidence('AnimateDiffPipeline()'), False)
Evidence.test_unbound = check_unbound

def check_wrong_provider(self):
    self.assertEqual(evidence('from unrelated import AnimateDiffPipeline\nAnimateDiffPipeline()'), False)
Evidence.test_wrong_provider = check_wrong_provider

def check_text_generation(self):
    self.assertEqual(evidence("from transformers import pipeline\npipeline('text-generation')"), False)
Evidence.test_text_generation = check_text_generation

def check_image_to_text(self):
    self.assertEqual(evidence("from transformers import pipeline\npipeline('image-to-text')"), False)
Evidence.test_image_to_text = check_image_to_text

def check_text_to_image(self):
    self.assertEqual(evidence("from transformers import pipeline\npipeline('text-to-image')"), False)
Evidence.test_text_to_image = check_text_to_image

def check_image_generation(self):
    self.assertEqual(evidence("from transformers import pipeline\npipeline('image-generation')"), False)
Evidence.test_image_generation = check_image_generation
