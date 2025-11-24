import zipfile
import fitz
#import olefile
import py7zr

class FileEncryptDetector:

    @staticmethod
    def is_pdf_encrypted(filepath):
        """
        返回:
        (bool, str)  
        bool = 是否加密  
        str  = 原因说明  
        """
        try:
            doc = fitz.open(filepath)
            if doc.is_encrypted:
                return True, "PDF 文件已加密，需要密码才能打开"
            return False, ""
        except Exception as e:
            return True, f"PDF 文件无法打开，可能已加密: {e}"

    @staticmethod
    def is_office_encrypted(filepath):
        """
        判断 docx/xlsx/pptx 是否加密
        """
        try:
            # docx/xlsx/pptx 本质是 ZIP
            with zipfile.ZipFile(filepath, 'r') as z:
                # 正常文件一定包含 [Content_Types].xml
                if '[Content_Types].xml' not in z.namelist():
                    return True, "Office 文件结构异常，可能已加密"
                return False, ""
        except RuntimeError as e:
            # 明确错误：Encrypted file
            if "encrypted" in str(e).lower():
                return True, "Office 文件被密码保护"
            return True, f"Office 文件无法读取，可能已加密: {e}"
        except zipfile.BadZipFile:
            return True, "Office 文件损坏或已加密"
        except Exception as e:
            return True, f"Office 文件无法读取，可能已加密: {e}"

    @staticmethod
    def is_zip_encrypted(filepath):
        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                for info in z.infolist():
                    # flag_bit 0x1 表示加密
                    if info.flag_bits & 0x1:
                        return True, "ZIP 包含加密文件"
            return False, ""
        except Exception:
            return False, ""
        
    @staticmethod
    def is_7z_encrypted(filepath):
        """
        判断 7z 是否加密。
        原理：尝试读取文件列表，如果是加密的，py7zr 会抛出 PasswordRequired 异常
        """
        try:
            with py7zr.SevenZipFile(filepath, mode='r') as archive:
                _ = archive.getnames()  # 读取内容列表
            return False,""
        except py7zr.exceptions.PasswordRequired:
            return True,"该7z文件为加密文件"
        except Exception:
            # 其他异常例如文件损坏，也可以认为不是标准的加密状态
            return False,""