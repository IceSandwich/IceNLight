import os
import typing

def GetIceNLightDir():
	return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def GetAssetsDir(subfolder: typing.Optional[str]):
	assetsDir = os.path.join(GetIceNLightDir(), "data", "assets")
	if subfolder is not None:
		assetsDir = os.path.join(assetsDir, subfolder)
	os.makedirs(assetsDir, exist_ok=True)
	return assetsDir