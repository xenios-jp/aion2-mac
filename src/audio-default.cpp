// Refresh Wine's default output from macOS and notify existing audio clients.
#include <windows.h>
#include <initguid.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    const bool apply = argc == 2 && !strcmp(argv[1], "--apply");
    if (argc > 1 && !apply) return 2;
    const wchar_t *key = L"Software\\Wine\\Drivers\\winecoreaudio.drv";
    // A fresh enumerator reads macOS's default only if no prior override remains.
    if (apply) {
        HKEY reset;
        if (!RegOpenKeyExW(HKEY_CURRENT_USER, key, 0, KEY_SET_VALUE, &reset)) {
            RegDeleteValueW(reset, L"DefaultOutput");
            RegCloseKey(reset);
        }
    }
    HRESULT init = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(init)) return 1;
    IMMDeviceEnumerator *enumerator = NULL;
    IMMDevice *device = NULL;
    IPropertyStore *properties = NULL;
    PROPVARIANT value;
    PropVariantInit(&value);
    HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, NULL,
        CLSCTX_INPROC_SERVER, IID_IMMDeviceEnumerator, (void **)&enumerator);
    if (SUCCEEDED(hr)) hr = enumerator->GetDefaultAudioEndpoint(eRender, eMultimedia, &device);
    if (SUCCEEDED(hr)) hr = device->OpenPropertyStore(STGM_READ, &properties);
    if (SUCCEEDED(hr)) hr = properties->GetValue(PKEY_Device_FriendlyName, &value);
    if (SUCCEEDED(hr) && value.vt == VT_LPWSTR) {
        char name[1024];
        if (WideCharToMultiByte(CP_UTF8, 0, value.pwszVal, -1, name, sizeof(name), NULL, NULL))
            printf("Wine default output: %s\n", name);
    }
    if (SUCCEEDED(hr) && apply) {
        LPWSTR id = NULL;
        hr = device->GetId(&id);
        if (SUCCEEDED(hr)) {
            HKEY output;
            LONG result = RegCreateKeyExW(HKEY_CURRENT_USER, key, 0, NULL, 0,
                KEY_SET_VALUE, NULL, &output, NULL);
            if (!result) {
                result = RegSetValueExW(output, L"DefaultOutput", 0, REG_SZ,
                    (BYTE *)id, (DWORD)((wcslen(id) + 1) * sizeof(wchar_t)));
                RegCloseKey(output);
            }
            printf("Windows default-output update status=%ld\n", result);
            if (result) hr = HRESULT_FROM_WIN32(result);
            CoTaskMemFree(id);
        }
    }
    if (FAILED(hr)) printf("Audio refresh HRESULT=%08lx\n", (unsigned long)hr);
    PropVariantClear(&value);
    if (properties) properties->Release();
    if (device) device->Release();
    if (enumerator) enumerator->Release();
    CoUninitialize();
    return FAILED(hr);
}
