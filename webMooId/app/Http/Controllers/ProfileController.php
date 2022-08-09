<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use App\Models\Peternakan;

class ProfileController extends Controller
{
    public function index()
    {
        $id = 1;
        $data_get = Peternakan::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'My Profile'
        );
        return view('peternak.pages.profile.index')->with($data);
    }

    public function edit()
    {
        $id = 1;
        $data_get = Peternakan::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'Update My Profile'
        );
        return view('peternak.pages.profile.form')->with($data);
    }

    public function update(Request $request)
    {
        $request->validate([
            'nama_peternakan' => 'required',
            'username' => 'required|unique:users',
            'password' => 'confirmed',
            'jenis_peternakan' => 'required',
            'nama_lengkap' => 'required',
            'no_telepon' => 'required|numeric',
            'email' => 'required|unique:users',
            'alamat_lengkap' => 'required',
            'latitude' => 'required',
            'longitude' => 'required',
            'area' => 'required|numeric',
        ]);

        $id = 1;
        $info = Peternakan::find($id);

        // Handle File Upload
        if ($request->hasFile('avatar')) {
            // Get just extension
            $avatar_extension = $request->file('avatar')->getClientOriginalExtension();
            // Filename to store
            $avatarNameToStore = 'avatar.'.$avatar_extension;
            // Check if file exists
            if (Storage::exists('peternakan/'.$info->username.'/'.$info->avatar)) {
                // Delete File
                Storage::delete('peternakan/'.$info->username.'/'.$info->avatar);
            }
            // Upload file
            $request->file('avatar')->storeAs('peternakan/'.$info->username.'/', $avatarNameToStore);
        }

        $info->nama_peternakan = $request->get('nama_peternakan');
        $info->username = $request->get('username');
        if (isset($info->password)) {
            $info->password = $request->get('password');
        }
        $info->nama_lengkap = $request->get('full_name');
        $info->no_telepon = $request->get('no_telepon');
        $info->email = $request->get('email');
        $info->alamat_lengkap = $request->get('alamat_lengkap');
        $info->latitude = $request->get('latitude');
        $info->longitude = $request->get('longitude');
        $info->area = $request->get('area');
        if ($request->hasFile('avatar')) {
            $info->avatar = $avatarNameToStore;
        }
        $info->save();

        return redirect(route('profile.index'))->with('success', 'Profile #'.$id.' berhasil diubah!');
    }
}
