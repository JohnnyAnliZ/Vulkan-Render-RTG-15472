# Vulkan renderer
My work on top of nakluV https://github.com/15-472/nakluV, the Vulkan tutorial for the class 15472 at CMU

# s72 Loader, s72 format to C++ data

# how to load a scene from blender(for windows):
run scenes\Makefile.WINDOWS with
$nmake -f makefile.windows
Change the content of the .WINDOWS file to change settings of running the exporter python script located in scenes\exporters

# go to the nakluV directory(For myself):
cd C:\Users\zal17\Documents\RealTimeG\nakluV

run viewer with a scene:
bin\main --scene scenes\examples\


tonemap specifiers:
bin\main --scene scenes\examples\materials.s72 --exposure 1 --tone-map REINHARD

# cubemap utility
convolving of lambertian and ggx split-sum cubemap environment images
bin\cube scenes\examples\ox_bridge_morning.png --lambertian scenes\examples\ox_bridge_morning_mine.png

