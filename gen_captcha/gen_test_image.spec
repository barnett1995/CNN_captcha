# -*- mode: python -*-

block_cipher = None


a = Analysis(['gen_test_image.py'],
             pathex=['/home/gxm/Documents/Git/CNN/gen_captcha'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='gen_test_image',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
