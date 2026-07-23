# Vulkan renderer
My work on top of nakluV https://github.com/15-472/nakluV, the Vulkan tutorial for the class 15472 at CMU

#Compilation
Node js is used compile the code. run $node maekfile.js to compile. For windows, the code is compiled using MSVC  

# How to load a scene from blender(for windows):
Run scenes\Makefile.WINDOWS with
$nmake -f makefile.windows
Change the content of the .WINDOWS file to change settings of running the exporter python script located in scenes\exporters
    
# Run viewer with a scene:
bin\main --scene scenes\examples\scene_name.s72


tonemap specifiers:
bin\main --scene scenes\examples\materials.s72 --exposure 1 --tone-map REINHARD

# cubemap utility
convolving of lambertian and ggx split-sum cubemap environment images
bin\cube scenes\examples\ox_bridge_morning.png --lambertian scenes\examples\ox_bridge_morning_mine.png

