<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Toko_Obat extends Model
{
    // Table Name
    protected $table = 'toko_obat';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'nama_toko_obat',
        'nama_lengkap',
        'email',
        'password',
        'no_telepon',
        'avatar',
    ];

    public function obat()
    {
        return $this->hasMany('App\Models\Obat','id_toko_obat');
    }
}
