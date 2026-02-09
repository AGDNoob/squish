; Squish NSIS Installer Script
; Build with: makensis installer.nsi

!include "MUI2.nsh"
!include "WinMessages.nsh"

; --------------------------------
; General
; --------------------------------
!define PRODUCT_NAME "Squish"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "AGDNoob"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "squish-${PRODUCT_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES\Squish"
InstallDirRegKey HKLM "Software\Squish" "InstallDir"
RequestExecutionLevel admin

; --------------------------------
; Interface Settings
; --------------------------------
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; --------------------------------
; Pages
; --------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; --------------------------------
; Languages
; --------------------------------
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "German"

; --------------------------------
; Installer Section
; --------------------------------
Section "Squish" SecMain
    SetOutPath "$INSTDIR"
    
    ; Copy files
    File "build\squish.exe"
    File "LICENSE"
    File "README.md"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"
    
    ; Add to system PATH
    ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
    StrCpy $0 "$0;$INSTDIR"
    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" $0
    ; Notify system of environment change
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
    
    ; Registry entries for Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "DisplayName" "Squish - Image Optimizer"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "DisplayIcon" "$INSTDIR\squish.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "Publisher" "Marlo"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                     "DisplayVersion" "1.0.0"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                       "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish" \
                       "NoRepair" 1
    
    ; Store install dir
    WriteRegStr HKLM "Software\Squish" "InstallDir" "$INSTDIR"
SectionEnd

; --------------------------------
; Uninstaller Section
; --------------------------------
Section "Uninstall"
    ; Note: PATH cleanup requires manual removal or EnvVarUpdate plugin
    ; For clean uninstall, user may need to remove $INSTDIR from PATH manually
    
    ; Delete files
    Delete "$INSTDIR\squish.exe"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\README.md"
    Delete "$INSTDIR\uninstall.exe"
    
    ; Remove directory
    RMDir "$INSTDIR"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Squish"
    DeleteRegKey HKLM "Software\Squish"
SectionEnd
